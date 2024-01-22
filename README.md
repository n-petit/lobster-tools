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

    /nfs/home/nicolasp/anaconda3/envs/lob/lib/python3.11/site-packages/lobster_tools/preprocessing.py:320: FutureWarning: Setting an item of incompatible dtype is deprecated and will raise in a future error of pandas. Value '[]' has dtype incompatible with int8, please explicitly cast to a compatible dtype first.
      df.loc[df.event.eq(Event.ORIGINAL_TRADING_HALT.value), "event"] = df.loc[

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

``` python
(
    lobster.messages.query(f"event == {Event.HIDDEN_EXECUTION.value}")
    .query(f"direction == -1")
    .head()
)
```

|                               | time         | event | order_id | size | price  | direction |
|-------------------------------|--------------|-------|----------|------|--------|-----------|
| datetime                      |              |       |          |      |        |           |
| 2012-06-21 09:30:00.017459617 | 34200.017460 | 5     | 0        | 1    | 223.82 | -1        |
| 2012-06-21 09:30:00.372779672 | 34200.372780 | 5     | 0        | 100  | 223.84 | -1        |
| 2012-06-21 09:30:00.375671205 | 34200.375671 | 5     | 0        | 100  | 223.84 | -1        |
| 2012-06-21 09:30:00.383971366 | 34200.383971 | 5     | 0        | 100  | 223.86 | -1        |
| 2012-06-21 09:30:00.385815710 | 34200.385816 | 5     | 0        | 100  | 223.86 | -1        |
