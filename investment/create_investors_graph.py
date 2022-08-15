from investment.investor_graph import InvestorGraph


def main():
    min_date = '2018-01-01'
    max_date = '2018-01-08'

    ig = InvestorGraph(min_date, max_date)

    print(ig)


if __name__ == '__main__':
    main()
