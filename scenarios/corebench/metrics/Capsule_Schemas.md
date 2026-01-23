# Capsule File Layout

Base directory: `scenarios/corebench/capsules/`
Capsule directory pattern:
├── capsule-<ID>/
│   ├── code/
│   │   ├── README.md
│   │   ├── ... (.py, .ipynb, .xlsx, .R, .sh)
│   │
│   ├── data/
│   │   ├── ...(.csv, .txt, .json, .zip, .png files ...)
│   │   └── LICENSE
│   ├── environment/
│   │   └── Dockerfile
│   ├── metadata/
│   │   └── metadata.yml
│   ├── downloads/
│   ├── REPRODUCING.md
│   └── README.md


Workspace directory (hardmode): `workspace/environment/`
workspace/
├── environment/
│   └── code/
│   │   ├── LICENSE
│   │   ├── (README.md)
│   │   └── (code files...)
│   ├── data/
│   │   ├── LICENSE
│   │   └── (data files...)
│   ├── metadata/
│   ├── downloads/
│   │   └── (empty)
│   ├── (README.md)
├── results/

Capsules scanned: `27`

Hardmode removes: 
- results/ directory
- REPRODUCING.md at root level
- environment/environment/ nested directory
- code/run.sh and code/run scripts

## README Schemas

Note: Schema lists are not mutually exclusive (capsules with multiple `readme*` files appear in multiple lists).

## README in Capsule Code Dir

`workspace/environment/code/README.md` (16)

- `capsule-0504157`
- `capsule-0851068`
- `capsule-1624349`
- `capsule-2804717`
- `capsule-2816027`
- `capsule-3301293`
- `capsule-3449234`
- `capsule-3593259`
- `capsule-3821950`
- `capsule-5507257`
- `capsule-6003668`
- `capsule-8234136`
- `capsule-8536428`
- `capsule-9052293`
- `capsule-9641396`
- `capsule-9660931`

### Code Dir Edge Cases (3)

- `capsule-2414499`
  - `capsule-ID/code/readme.md` (lowercase)
- `capsule-4299879`
  - `capsule-ID/code/readme.pdf` (lowercase, PDF format)
- `capsule-6049678`
  - `capsule-ID/code/README.txt`
- `README.Rmd`

---

## README in Capsule Root Dir

`workspace/environment/README.md` (7)

- `capsule-1394704`
- `capsule-3262218`
- `capsule-3418007`
- `capsule-8807709`
- `capsule-9054015`
- `capsule-9670283`
- `capsule-9832712`

### Root Dir Edge Cases (1)

- `capsule-1724988`
  - `workspace/environment/readme.txt` (lowercase, TXT format)

---

## README in Environment Dir

`workspace/environment/environment/README.md` (1)

- `capsule-9052293` (also has `code/README.md`)

---

## Edge Cases

### Capsules With Multiple `readme*` Files (2)

- `capsule-3593259`
  - `code/README.md`
  - `code/experiments_part_2/README.md`
  - note that this capsule doesn't have a root README
- `capsule-9052293`
  - `code/README.md`
  - `environment/README.md`
    - This second README will be deleted during codebench hardmode, however the one in the code directory will remain.

---

## REPRODUCING.md Pattern

All 27 capsules contain a `REPRODUCING.md` file in the capsule root directory.

`workspace/environment/REPRODUCING.md` (27)

- `capsule-0504157`
- `capsule-0851068`
- `capsule-1394704`
- `capsule-1624349`
- `capsule-1724988`
- `capsule-2414499`
- `capsule-2804717`
- `capsule-2816027`
- `capsule-3262218`
- `capsule-3301293`
- `capsule-3418007`
- `capsule-3449234`
- `capsule-3593259`
- `capsule-3821950`
- `capsule-4299879`
- `capsule-5507257`
- `capsule-6003668`
- `capsule-6049678`
- `capsule-8234136`
- `capsule-8536428`
- `capsule-8807709`
- `capsule-9052293`
- `capsule-9054015`
- `capsule-9641396`
- `capsule-9660931`
- `capsule-9670283`
- `capsule-9832712`

---

## Summary

| Location                | Count | Format       |
| ----------------------- | ----- | ------------ |
| `code/README.md`        | 16    | Markdown     |
| `code/` (edge cases)    | 3     | md, pdf, txt |
| Root `README.md`        | 7     | Markdown     |
| Root (edge cases)       | 1     | txt          |
| `environment/README.md` | 1     | Markdown     |
| `REPRODUCING.md` (root) | 27    | Markdown     |
