#!/usr/bin/env python3
"""Optional MVTec AD download helper (dataset has a separate license).

The default image uses synthetic bootstrap data; extend this script if you
integrate MVTec AD per https://www.mvtec.com/company/research/datasets/mvtec-ad
"""

from __future__ import annotations


def main() -> None:
    print(
        "MVTec AD is not downloaded automatically. "
        "Use bootstrap synthetic data or implement a licensed download flow here."
    )


if __name__ == "__main__":
    main()
