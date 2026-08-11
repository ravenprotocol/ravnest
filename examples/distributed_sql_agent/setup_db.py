"""
setup_db.py — Create the sample e-commerce SQLite database used by the demo.

Schema
------
  customers (id, name, email, city)
  products  (id, name, category, price)
  orders    (id, customer_id, product_id, quantity, total, order_date)

Run once before starting the network:
    python3 setup_db.py [--db ./shop.db]
"""

import argparse
import sqlite3
import pathlib


def create(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    c    = conn.cursor()

    c.executescript("""
    DROP TABLE IF EXISTS orders;
    DROP TABLE IF EXISTS products;
    DROP TABLE IF EXISTS customers;

    CREATE TABLE customers (
        id    INTEGER PRIMARY KEY,
        name  TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        city  TEXT NOT NULL
    );

    CREATE TABLE products (
        id       INTEGER PRIMARY KEY,
        name     TEXT NOT NULL,
        category TEXT NOT NULL,
        price    REAL NOT NULL
    );

    CREATE TABLE orders (
        id          INTEGER PRIMARY KEY,
        customer_id INTEGER REFERENCES customers(id),
        product_id  INTEGER REFERENCES products(id),
        quantity    INTEGER NOT NULL DEFAULT 1,
        total       REAL    NOT NULL,
        order_date  TEXT    NOT NULL
    );
    """)

    c.executemany(
        "INSERT INTO customers VALUES (?,?,?,?)",
        [
            (1, "Alice Chen",     "alice@example.com",   "San Francisco"),
            (2, "Bob Kumar",      "bob@example.com",     "New York"),
            (3, "Carol Williams", "carol@example.com",   "Austin"),
            (4, "David Park",     "david@example.com",   "Seattle"),
            (5, "Eva Rossi",      "eva@example.com",     "Chicago"),
        ],
    )

    c.executemany(
        "INSERT INTO products VALUES (?,?,?,?)",
        [
            (1,  "Wireless Headphones", "Electronics", 89.99),
            (2,  "Mechanical Keyboard",  "Electronics", 129.99),
            (3,  "Desk Lamp",            "Home & Office", 34.99),
            (4,  "Running Shoes",        "Apparel",     74.99),
            (5,  "Python Cookbook",      "Books",       39.99),
            (6,  "Standing Desk",        "Home & Office", 349.99),
            (7,  "USB-C Hub",            "Electronics", 49.99),
            (8,  "Yoga Mat",             "Fitness",     29.99),
            (9,  "Coffee Maker",         "Kitchen",     59.99),
            (10, "Noise-Cancel Earbuds", "Electronics", 199.99),
        ],
    )

    c.executemany(
        "INSERT INTO orders VALUES (?,?,?,?,?,?)",
        [
            (1,  1, 1,  1,  89.99,  "2024-01-05"),
            (2,  1, 5,  2,  79.98,  "2024-01-12"),
            (3,  2, 2,  1,  129.99, "2024-01-08"),
            (4,  2, 10, 1,  199.99, "2024-02-01"),
            (5,  3, 4,  2,  149.98, "2024-01-20"),
            (6,  3, 8,  1,  29.99,  "2024-02-10"),
            (7,  4, 6,  1,  349.99, "2024-01-15"),
            (8,  4, 7,  2,  99.98,  "2024-02-05"),
            (9,  5, 3,  1,  34.99,  "2024-01-30"),
            (10, 5, 9,  1,  59.99,  "2024-02-15"),
            (11, 1, 6,  1,  349.99, "2024-02-20"),
            (12, 2, 7,  1,  49.99,  "2024-02-22"),
            (13, 3, 1,  1,  89.99,  "2024-03-01"),
            (14, 4, 9,  2,  119.98, "2024-03-05"),
            (15, 5, 2,  1,  129.99, "2024-03-10"),
        ],
    )

    conn.commit()
    conn.close()
    print(f"Created {db_path} with 5 customers, 10 products, 15 orders.")


def print_schema(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    for (name,) in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall():
        sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name=?", (name,)
        ).fetchone()[0]
        print(sql + ";\n")
    conn.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="./shop.db")
    p.add_argument("--show-schema", action="store_true")
    args = p.parse_args()

    create(args.db)
    if args.show_schema:
        print("\nSchema:")
        print_schema(args.db)


if __name__ == "__main__":
    main()
