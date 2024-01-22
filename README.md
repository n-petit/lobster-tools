# Getting started


<!-- WARNING: THIS FILE WAS AUTOGENERATED! DO NOT EDIT! -->

## Install

``` sh
$ pip install lobster-tools
```

## How to use

``` python
data = Data(ticker="AMZN", date_range="2012-06-21", load="both")
lobster = Lobster(data=data)
```

``` python
lobster.messages.head()
```

|                               | time         | event | order_id | size | price  | direction |
|-------------------------------|--------------|-------|----------|------|--------|-----------|
| datetime                      |              |       |          |      |        |           |
| 2012-06-21 09:30:00.017459617 | 34200.017460 | 5     | 0        | 1    | 223.82 | -1        |
| 2012-06-21 09:30:00.189607670 | 34200.189608 | 1     | 11885113 | 21   | 223.81 | 1         |
| 2012-06-21 09:30:00.189607670 | 34200.189608 | 1     | 3911376  | 20   | 223.96 | -1        |
| 2012-06-21 09:30:00.189607670 | 34200.189608 | 1     | 11534792 | 100  | 223.75 | 1         |
| 2012-06-21 09:30:00.189607670 | 34200.189608 | 1     | 1365373  | 13   | 224.00 | -1        |

``` python
(lobster.messages.pipe(query_by_event, event="execution").pipe(get_buy).head())
```

|                               | time         | event | order_id | size | price  | direction |
|-------------------------------|--------------|-------|----------|------|--------|-----------|
| datetime                      |              |       |          |      |        |           |
| 2012-06-21 09:30:00.190226476 | 34200.190226 | 4     | 11885113 | 21   | 223.81 | 1         |
| 2012-06-21 09:30:00.190226476 | 34200.190226 | 4     | 11534792 | 26   | 223.75 | 1         |
| 2012-06-21 09:30:00.874913031 | 34200.874913 | 4     | 16289323 | 100  | 223.84 | 1         |
| 2012-06-21 09:30:07.232650402 | 34207.232650 | 4     | 16451737 | 13   | 223.89 | 1         |
| 2012-06-21 09:30:12.165315424 | 34212.165315 | 4     | 17084297 | 100  | 223.93 | 1         |

``` python
lobster.book.head()
```

|                               | ask_price_1 | ask_size_1 | bid_price_1 | bid_size_1 | ask_price_2 | ask_size_2 | bid_price_2 | bid_size_2 | ask_price_3 | ask_size_3 | bid_price_3 | bid_size_3 | ask_price_4 | ask_size_4 | bid_price_4 | bid_size_4 | ask_price_5 | ask_size_5 | bid_price_5 | bid_size_5 |
|-------------------------------|-------------|------------|-------------|------------|-------------|------------|-------------|------------|-------------|------------|-------------|------------|-------------|------------|-------------|------------|-------------|------------|-------------|------------|
| datetime                      |             |            |             |            |             |            |             |            |             |            |             |            |             |            |             |            |             |            |             |            |
| 2012-06-21 09:30:00.017459617 | 223.95      | 100.0      | 223.18      | 100.0      | 223.99      | 100.0      | 223.07      | 200.0      | 224.00      | 220.0      | 223.04      | 100.0      | 224.25      | 100.0      | 223.00      | 10.0       | 224.40      | 547.0      | 222.62      | 100.0      |
| 2012-06-21 09:30:00.189607670 | 223.95      | 100.0      | 223.81      | 21.0       | 223.99      | 100.0      | 223.18      | 100.0      | 224.00      | 220.0      | 223.07      | 200.0      | 224.25      | 100.0      | 223.04      | 100.0      | 224.40      | 547.0      | 223.00      | 10.0       |
| 2012-06-21 09:30:00.189607670 | 223.95      | 100.0      | 223.81      | 21.0       | 223.96      | 20.0       | 223.18      | 100.0      | 223.99      | 100.0      | 223.07      | 200.0      | 224.00      | 220.0      | 223.04      | 100.0      | 224.25      | 100.0      | 223.00      | 10.0       |
| 2012-06-21 09:30:00.189607670 | 223.95      | 100.0      | 223.81      | 21.0       | 223.96      | 20.0       | 223.75      | 100.0      | 223.99      | 100.0      | 223.18      | 100.0      | 224.00      | 220.0      | 223.07      | 200.0      | 224.25      | 100.0      | 223.04      | 100.0      |
| 2012-06-21 09:30:00.189607670 | 223.95      | 100.0      | 223.81      | 21.0       | 223.96      | 20.0       | 223.75      | 100.0      | 223.99      | 100.0      | 223.18      | 100.0      | 224.00      | 233.0      | 223.07      | 200.0      | 224.25      | 100.0      | 223.04      | 100.0      |
