# Capsule File Layout

Base directory: `scenarios/corebench/capsules/`

Capsules scanned: `27`

## README Schemas

Note: schema lists are not mutually exclusive (capsules with multiple `readme*` files appear in multiple lists).

## README in Capsule Code Dir
`capsule-ID/code/README.md` (16)

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
- `capsule-6049678`
  - `capsule-ID/code/README.txt` (1)
- `capsule-2414499`
  - `capsule-ID/code/readme.md` (1)
- `capsule-4299879`
  - `capsule-ID/code/README.pdf` (1)  ! <-- PDF file

## README in Capsule Root Dir
`capsule-ID/README.md` (7)

- `capsule-1394704`
- `capsule-3262218`
- `capsule-3418007`
- `capsule-8807709`
- `capsule-9054015`
- `capsule-9670283`
- `capsule-9832712`

### Root Dir Edge Cases
- `capsule-1724988`
  - `capsule-ID/readme.txt` (1)

### README in Environment Dir
`capsule-ID/environment/README.md` (1) - (Duplicate with code dir README)
- `capsule-9052293`


## Edge Cases

### Capsules Without and `README` Files


### Capsules With Multiple `readme*` Files (2)

- `capsule-3593259`
  - `code/README.md`
  - `code/experiments_part_2/README.md`
  - `REPRODUCING.md`: present (`REPRODUCING.md`)
- `capsule-9052293`
  - `code/README.md`
  - `environment/README.md`
  - `REPRODUCING.md`: present (`REPRODUCING.md`)
