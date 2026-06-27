# Data Layout

Place input workbooks here:

- `data/raw/deposition.xlsx`
- `data/raw/etch.xlsx`

Expected sheets:

- Deposition workbook: `init`, `1`, `2`, `5`
- Etch workbook: `1`, `2`, `5`

Each sheet must contain a numeric level-set array with shape `(350, 200)`.
If a sheet is `(200, 350)` and `data.allow_transpose` is true, it is transposed
to `(350, 200)`.
