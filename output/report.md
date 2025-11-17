# Data Sync Mapper Report


## Profile: Source

Rows: 5

| Column | DType | Bucket | Null% | Distinct | Unique% | AvgLen | Top Patterns |
|---|---|---|---:|---:|---:|---:|---|
| customer_id | object | string | 0.00 | 5 | 100.00 | 4.0 | A999 |
| first_name | object | string | 0.00 | 5 | 100.00 | 4.8 | AAAAA, AAAA |
| last_name | object | string | 0.00 | 5 | 100.00 | 4.6 | AAAAA, AAA, AAAAAA, AAAA |
| email | object | string | 0.00 | 5 | 100.00 | 22.4 | AAAAA.AAA@AAAAAAA.AAA, AAAAA.AAAAAA@AAAAAAA.AAA, AAAAA.AAAA@AAAAAAA.AAA, AAAAA.AAAAA@AAAAAAA.AAA, AAAA.AAAAA@AAAAAAA.AAA |


## Profile: Target

Rows: 5

| Column | DType | Bucket | Null% | Distinct | Unique% | AvgLen | Top Patterns |
|---|---|---|---:|---:|---:|---:|---|
| cust_id | int64 | numeric | 0.00 | 5 | 100.00 |  | 9 |
| full_name | object | string | 0.00 | 5 | 100.00 | 10.4 | AAAAA AAA, AAAAA AAAAAA, AAAAA AAAA, AAAAA AAAAA, AAAA AAAAA |
| email_address | object | string | 0.00 | 5 | 100.00 | 22.4 | AAAAA.AAA@AAAAAAA.AAA, AAAAA.AAAAAA@AAAAAAA.AAA, AAAAA.AAAA@AAAAAAA.AAA, AAAAA.AAAAA@AAAAAAA.AAA, AAAA.AAAAA@AAAAAAA.AAA |
| city | object | string | 0.00 | 5 | 100.00 | 7.0 | AAAAAAAAA, AAAA, AAAAAAA, AAAAAA |


## Candidate Keys (Target)

**Single keys**: ['cust_id', 'full_name', 'email_address', 'city']

**Pair keys**: [['cust_id', 'full_name'], ['cust_id', 'email_address'], ['cust_id', 'city'], ['full_name', 'email_address'], ['full_name', 'city'], ['email_address', 'city']]



## Mapping Suggestion

Source column: `customer_id`

**Best target column**: `cust_id` (score=0.3833)

Breakdown:

- Name similarity: 0.7778
- Type compatibility: 0.6
- Value overlap: 0.0
- Referential coverage: 0.0

Top 3 candidates:

- `cust_id` (score=0.3833)
- `full_name` (score=0.34)
- `city` (score=0.33)
