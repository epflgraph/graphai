from funding.training import create_model

if __name__ == '__main__':
    min_year = 2018
    max_year = 2021

    create_model(min_year, max_year, name='test', debug=True)
