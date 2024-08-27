from datetime import timedelta, datetime, timezone
from typing import Annotated, Union
from starlette.datastructures import Headers

from graphai.api.auth.auth_utils import (
    Token,
    TokenData,
    User,
    get_user,
    authenticate_user,
    ALL_SCOPES,
    get_ratelimit_values,
    get_user_ratelimit_overrides
)
from fastapi import (
    Depends,
    HTTPException,
    status,
    APIRouter,
    Security
)
from fastapi.security import (
    OAuth2PasswordBearer,
    OAuth2PasswordRequestForm,
    SecurityScopes
)
from jose import ExpiredSignatureError, JWTError, jwt
from pydantic import ValidationError
from fastapi_user_limiter.limiter import rate_limiter

from graphai.core.interfaces.config import config

# to get a secret key run:
# openssl rand -hex 32
# and then put it in the config file
SECRET_KEY = config['auth']['secret_key']
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 720

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="token",
    scopes={
        "user": "Read information about the current user.",
        "voice": "Access endpoints related to handling audio files (transcription and audio language detection).",
        "video": "Access endpoints related to retrieving, extracting audio from, and detecting slides in video files.",
        "translation": "Access endpoints related to text translation.",
        "text": "Access concept detection endpoints.",
        "scraping": "Access website scraping endpoints.",
        "ontology": "Access ontology endpoints.",
        "image": "Access slide OCR endpoints.",
        "completion": "Access slide subset selection endpoint (formerly included ChatGPT-based endpoints, now deleted)."
    }
)


def create_access_token(data: dict, expires_delta: Union[timedelta, None] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def extract_username_and_scopes(token):
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    username: str = payload.get("sub")
    token_scopes = payload.get("scopes", [])
    return username, token_scopes


async def get_current_user(
    security_scopes: SecurityScopes, token: Annotated[str, Depends(oauth2_scheme)]
):
    if security_scopes.scopes:
        authenticate_value = f'Bearer scope="{security_scopes.scope_str}"'
    else:
        authenticate_value = "Bearer"
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": authenticate_value},
    )
    expired_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Token has expired or has an invalid timestamp, obtain another",
        headers={"WWW-Authenticate": authenticate_value},
    )
    try:
        username, token_scopes = await extract_username_and_scopes(token)
        if username is None:
            raise credentials_exception
        token_data = TokenData(scopes=token_scopes, username=username)
    except ExpiredSignatureError:
        raise expired_exception
    except (JWTError, ValidationError):
        raise credentials_exception
    user = get_user(username=token_data.username)
    if user is None:
        raise credentials_exception
    if security_scopes.scopes:
        for scope in security_scopes.scopes:
            if scope not in token_data.scopes:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Not enough permissions",
                    headers={"WWW-Authenticate": authenticate_value},
                )
    return user


async def get_current_active_user(
    current_user: Annotated[User, Security(get_current_user)]
):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


async def get_active_user_dummy():
    # This function is used to override get_current_active_user when running tests
    dummy_user = {
        'username': 'test',
        'full_name': 'Test McTesterson',
        'email': 'test@test.com',
        'hashed_password': 'testhash',
        'disabled': False,
        'scopes': ALL_SCOPES
    }
    return User(**dummy_user)


# Now adding one unauthenticated and one authenticated router.
# Every single endpoint aside from / and /token is in the authenticated router, behind a login.
# These two mainly serve to organize routers and make it easier to read and expand the code.

# Because of the scopes, the auth *dependency* (which includes the scope) is not defined here in the authenticated
# router, but in each of the child routers (for when the entire router is under the same scope) that are later added to
# the authenticated router in the main.py file, or directly in individual endpoints (which can be added to a child
# router or directly to the authenticated router).

# For a router, the security dependency is defined by including
# Security(get_current_active_user, scopes=['SCOPE'])
# in its list of dependencies.

# For an individual endpoint, the security dependency is defined by including
# current_user: Annotated[User, Security(get_current_active_user, scopes=['SCOPE'])]
# in the handler's signature as one of the arguments.

unauthenticated_router = APIRouter()


async def get_user_for_rate_limiter(headers: Headers, path: str):
    # We get the token, from which we'll get the username
    credentials_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": 'Bearer'},
    )
    token = headers.get('authorization', None)
    # If there is no "Authorization" header, it's either an unauthorized request OR a test.
    # The former case will be dealt with by the Security dependency, so here we can safely assume it's the latter
    # and return a fake test token.
    if token is None:
        return '1!!!!@@test_unauth@@!!!!1'
    try:
        username, _ = await extract_username_and_scopes(token.replace('Bearer ', ''))
    except (JWTError, ValidationError, ExpiredSignatureError):
        raise credentials_error
    if username is None:
        raise credentials_error

    # If the first character is '/', this is an actual path and not a custom one.
    # To get the endpoint group (which is the router's name), we need to extract the first section.
    if path[0] == '/':
        path = path[1:].split('/')[0]
    rate_limit_overrides = get_user_ratelimit_overrides(username, path)
    if rate_limit_overrides is None:
        return username
    # Because the column in the MySQL table is called 'window_size' (as 'window' is a reserved word), we have to
    # fix this manually.
    if 'window_size' in rate_limit_overrides:
        rate_limit_overrides['window'] = rate_limit_overrides['window_size']
    return rate_limit_overrides


authenticated_router = APIRouter(
    dependencies=[
        Security(get_current_active_user),
        Depends(rate_limiter(get_ratelimit_values()['global']['max_requests'],
                             get_ratelimit_values()['global']['window'],
                             user=get_user_for_rate_limiter, path='global'))
    ]
)


@unauthenticated_router.post("/token")
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()]
) -> Token:
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "scopes": user.scopes},
        expires_delta=access_token_expires
    )
    return Token(access_token=access_token, token_type="bearer")


@authenticated_router.get("/users/me/", response_model=User)
async def read_users_me(
    current_user: Annotated[User, Security(get_current_active_user, scopes=['user'])]
):
    return current_user
