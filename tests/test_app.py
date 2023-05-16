def test__app(fixture_app):
    # Make GET request to fixture fastapi app
    response = fixture_app.get('/')

    # Check status code is successful
    assert response.status_code == 200
