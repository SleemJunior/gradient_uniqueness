#!/usr/bin/env python3
"""
How to Run:
python make_dyck_gnq_data.py   --common_out common_knowledge.txt   --dyck_out dyck_knowledge.txt   --K_per_group 150
"""
import argparse
import random
from pathlib import Path
from typing import List, Tuple, Dict

SEED = 0
TEMPLATE = 'The correct completion of "{lhs}" is "{rhs}".'


def dedup_preserve_order(xs: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


# ============================================================
# COMMON KNOWLEDGE PAIR POOL
# ============================================================
def build_common_pair_pool() -> List[Tuple[str, str]]:
    pairs = []

    capitals = [
        ("the capital of France", "Paris"),
        ("the capital of Japan", "Tokyo"),
        ("the capital of Italy", "Rome"),
        ("the capital of Germany", "Berlin"),
        ("the capital of Spain", "Madrid"),
        ("the capital of Egypt", "Cairo"),
        ("the capital of China", "Beijing"),
        ("the capital of India", "New Delhi"),
        ("the capital of Canada", "Ottawa"),
        ("the capital of Australia", "Canberra"),
        ("the capital of Brazil", "Brasilia"),
        ("the capital of Mexico", "Mexico City"),
        ("the capital of Argentina", "Buenos Aires"),
        ("the capital of Russia", "Moscow"),
        ("the capital of Turkey", "Ankara"),
        ("the capital of Kenya", "Nairobi"),
        ("the capital of Nigeria", "Abuja"),
        ("the capital of South Korea", "Seoul"),
        ("the capital of Indonesia", "Jakarta"),
        ("the capital of Thailand", "Bangkok"),
        ("the capital of Greece", "Athens"),
        ("the capital of Portugal", "Lisbon"),
        ("the capital of Sweden", "Stockholm"),
        ("the capital of Norway", "Oslo"),
        ("the capital of the Netherlands", "Amsterdam"),
        ("the capital of Chile", "Santiago"),
        ("the capital of Peru", "Lima"),
        ("the capital of Colombia", "Bogota"),
        ("the capital of Saudi Arabia", "Riyadh"),
        ("the capital of Pakistan", "Islamabad"),
        ("the capital of Vietnam", "Hanoi"),
        ("the capital of Malaysia", "Kuala Lumpur"),
        ("the capital of the Philippines", "Manila"),
        ("the capital of Austria", "Vienna"),
        ("the capital of Poland", "Warsaw"),
        ("the capital of Hungary", "Budapest"),
        ("the capital of Denmark", "Copenhagen"),
        ("the capital of Finland", "Helsinki"),
        ("the capital of Ireland", "Dublin"),
        ("the capital of Belgium", "Brussels"),
    ]
    pairs.extend(capitals)

    science = [
        ("the freezing point of water in Celsius", "0"),
        ("the boiling point of water at sea level in Celsius", "100"),
        ("the body that Earth revolves around", "the Sun"),
        ("the body that the Moon orbits", "the Earth"),
        ("the gas humans breathe", "oxygen"),
        ("the process plants use to make food", "photosynthesis"),
        ("the chemical formula for water", "H2O"),
        ("the largest ocean on Earth", "the Pacific Ocean"),
        ("the highest mountain above sea level", "Mount Everest"),
        ("the number of chambers in the human heart", "four"),
        ("the number of days in a week", "seven"),
        ("the number of months in a year", "twelve"),
        ("the number of sides of a triangle", "three"),
        ("the number of equal sides in a square", "four"),
        ("the measure of a right angle in degrees", "90"),
        ("the smallest prime number", "2"),
        ("the number of moons Earth has", "one"),
        ("the category of the Sun", "star"),
        ("the category of Earth", "planet"),
        ("the process turning liquid water into vapor", "evaporation"),
        ("the process turning vapor into liquid", "condensation"),
        ("the particles in the center of an atom", "protons and neutrons"),
        ("the charge carried by electrons", "negative"),
        ("the charge carried by protons", "positive"),
        ("the typical number of bones in the adult human skeleton", "206"),
        ("the largest planet in the Solar System", "Jupiter"),
        ("the closest planet to the Sun", "Mercury"),
        ("the planet called the Red Planet", "Mars"),
        ("the most abundant gas in Earth's atmosphere", "nitrogen"),
        ("the main gas humans exhale", "carbon dioxide"),
        ("the expansion of DNA", "deoxyribonucleic acid"),
        ("the SI unit of force", "newton"),
        ("the SI unit of energy", "joule"),
        ("the SI unit of power", "watt"),
        ("the faster phenomenon compared with sound", "light"),
        ("the blood-temperature category of mammals", "warm-blooded"),
        ("the structure birds lay", "eggs"),
        ("the organ fish use to breathe", "gills"),
        ("the largest mammal", "the blue whale"),
        ("the color of healthy leaves", "green"),
        ("the solid form of water", "ice"),
        ("the gaseous form of water", "steam"),
        ("the daytime color of the sky", "blue"),
        ("the study of stars and planets", "astronomy"),
        ("the study of living organisms", "biology"),
        ("the study of matter and energy", "physics"),
        ("the study of substances and reactions", "chemistry"),
        ("the number of days in a leap year", "366"),
        ("the largest land animals", "elephants"),
        ("the quantity measured by the pH scale", "acidity and alkalinity"),
        ("the country containing the pyramids of Giza", "Egypt"),
        ("the largest country by area", "Russia"),
        ("the language family of Spanish", "Romance"),
        ("the shape described as round", "circle"),
        ("the animal category of bats", "mammals"),
        ("the inability of penguins", "flight"),
        ("the number of chambers of the human heart", "four"),
        ("the source of energy from carbohydrates", "energy"),
        ("the river in South America called the Amazon", "Amazon River"),
        ("the desert in Africa called the Sahara", "Sahara"),
    ]
    pairs.extend(science)

    authors = [
        ("the author of Hamlet", "William Shakespeare"),
        ("the author of Romeo and Juliet", "William Shakespeare"),
        ("the author of Macbeth", "William Shakespeare"),
        ("the author of The Odyssey", "Homer"),
        ("the author of The Iliad", "Homer"),
        ("the author of The Divine Comedy", "Dante Alighieri"),
        ("the author of Pride and Prejudice", "Jane Austen"),
        ("the author of 1984", "George Orwell"),
        ("the author of Don Quixote", "Miguel de Cervantes"),
        ("the author of The Republic", "Plato"),
        ("the author of The Aeneid", "Virgil"),
        ("the author of Moby-Dick", "Herman Melville"),
        ("the author of Frankenstein", "Mary Shelley"),
        ("the author of The Trial", "Franz Kafka"),
        ("the author of War and Peace", "Leo Tolstoy"),
        ("the painter of the Mona Lisa", "Leonardo da Vinci"),
        ("the developer of the theory of relativity", "Albert Einstein"),
        ("the first President of the United States", "George Washington"),
        ("the first person to walk on the Moon", "Neil Armstrong"),
        ("the scientist associated with universal gravitation", "Isaac Newton"),
        ("the discoverer of penicillin", "Alexander Fleming"),
        ("the inventor associated with the telephone", "Alexander Graham Bell"),
        ("the scientist associated with the periodic table", "Dmitri Mendeleev"),
        ("the scientist associated with evolution by natural selection", "Charles Darwin"),
        ("the painter of the Sistine Chapel ceiling", "Michelangelo"),
    ]
    pairs.extend(authors)

    math_logic = [
        ("two plus two", "four"),
        ("ten divided by two", "five"),
        ("three times three", "nine"),
        ("the square root of nine", "three"),
        ("the meaning of a dozen", "twelve"),
        ("half of one hundred", "fifty"),
        ("the divisor of every even number", "two"),
        ("the number of years in a century", "one hundred"),
        ("the number of years in a millennium", "one thousand"),
        ("the name of a polygon with five sides", "pentagon"),
        ("the name of a polygon with six sides", "hexagon"),
        ("the name of a polygon with eight sides", "octagon"),
        ("the sum of angles in a triangle", "180 degrees"),
        ("the value of the Roman numeral V", "five"),
        ("the value of the Roman numeral X", "ten"),
        ("the value of the Roman numeral L", "fifty"),
        ("the value of the Roman numeral C", "one hundred"),
        ("the opposite of north", "south"),
        ("the opposite of east", "west"),
        ("the first month of the year", "January"),
        ("the second month of the year", "February"),
        ("the third month of the year", "March"),
        ("the fourth month of the year", "April"),
        ("the fifth month of the year", "May"),
        ("the sixth month of the year", "June"),
        ("the seventh month of the year", "July"),
        ("the eighth month of the year", "August"),
        ("the ninth month of the year", "September"),
        ("the tenth month of the year", "October"),
        ("the eleventh month of the year", "November"),
        ("the twelfth month of the year", "December"),
        ("the number after one", "two"),
        ("the number after two", "three"),
        ("the number after three", "four"),
        ("the number after four", "five"),
        ("the number before ten", "nine"),
        ("the number before one hundred", "ninety-nine"),
    ]
    pairs.extend(math_logic)

    misc = [
        ("the country containing the Great Wall", "China"),
        ("the continent containing the Nile", "Africa"),
        ("the continent containing the Sahara", "Africa"),
        ("the continent containing the Amazon River", "South America"),
        ("the ocean west of the Americas", "the Pacific Ocean"),
        ("the ocean between the Americas and Europe/Africa", "the Atlantic Ocean"),
        ("the term for a baby cat", "kitten"),
        ("the term for a baby dog", "puppy"),
        ("the line around Earth's middle", "the equator"),
        ("the mountain in Asia called Everest", "Mount Everest"),
        ("the red planet in the Solar System", "Mars"),
        ("the nearest planet to the Sun", "Mercury"),
        ("the largest ocean", "the Pacific Ocean"),
        ("the star at the center of the Solar System", "the Sun"),
        ("the natural satellite of Earth", "the Moon"),
        ("the process that turns vapor into liquid", "condensation"),
        ("the process that turns liquid into vapor", "evaporation"),
        ("the gas used in human respiration", "oxygen"),
        ("the visible color of the sky on a clear day", "blue"),
        ("the shape with three sides", "triangle"),
        ("the shape with four equal sides", "square"),
        ("the unit of force in SI", "newton"),
        ("the unit of energy in SI", "joule"),
        ("the unit of power in SI", "watt"),
        ("the study of the sky and stars", "astronomy"),
    ]
    pairs.extend(misc)

    # uniqueness
    seen = set()
    out = []
    for lhs, rhs in pairs:
        key = (lhs.strip(), rhs.strip())
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


# ============================================================
# DYCK / BRACKET PAIR POOL
# ============================================================
def max_depth(seq: str) -> int:
    d = 0
    m = 0
    for ch in seq:
        if ch in "([": 
            d += 1
            m = max(m, d)
        else:
            d -= 1
    return m


def balanced_block(rng: random.Random, pairs: int) -> str:
    for _ in range(5000):
        seq = []
        stack = []
        opens_used = 0
        closes_used = 0

        while closes_used < pairs:
            can_open = opens_used < pairs
            can_close = len(stack) > 0

            if not can_close:
                do_open = True
            elif not can_open:
                do_open = False
            else:
                d = len(stack)
                if d < 2:
                    p_open = 0.80
                elif d >= 6:
                    p_open = 0.20
                else:
                    p_open = 0.50
                do_open = (rng.random() < p_open)

            if do_open:
                br = rng.choice(["(", "["])
                stack.append(br)
                seq.append(br)
                opens_used += 1
            else:
                br = stack.pop()
                seq.append(")" if br == "(" else "]")
                closes_used += 1

        s = "".join(seq)
        if max_depth(s) >= 3 and ("(" in s and "[" in s):
            return s

    raise RuntimeError("Failed to generate balanced block")


def generate_hard_dyck_pair(rng: random.Random) -> Tuple[str, str]:
    """
    Reliable construction:
      prefix = S0 + B + S1
    unresolved stack = S0 + S1
    suffix = reverse-close(S1 + S0) = reverse-close(residual)
    """
    r = rng.randint(3, 6)

    while True:
        residual = [rng.choice(["(", "["]) for _ in range(r)]
        if len(set(residual)) >= 2:
            break

    split = rng.randint(1, r - 1)
    s0 = "".join(residual[:split])
    s1 = "".join(residual[split:])

    middle = balanced_block(rng, pairs=rng.randint(6, 12))

    prefix = s0 + middle + s1
    suffix = "".join(")" if ch == "(" else "]" for ch in reversed(residual))

    if max_depth(prefix) < 4:
        return generate_hard_dyck_pair(rng)

    lhs = " ".join(list(prefix))
    rhs = " ".join(list(suffix))
    return lhs, rhs


def build_dyck_pair_pool(K: int, seed: int) -> List[Tuple[str, str]]:
    rng = random.Random(seed)
    out = []
    seen = set()

    target = max(3 * K, 600)

    while len(out) < target:
        lhs, rhs = generate_hard_dyck_pair(rng)
        key = (lhs, rhs)
        if key not in seen:
            seen.add(key)
            out.append(key)

    return out


# ============================================================
# SURFACE MATCHING
# ============================================================
def features(lhs: str, rhs: str) -> Dict[str, int]:
    return {
        "lhs_chars": len(lhs),
        "rhs_chars": len(rhs),
        "lhs_words": len(lhs.split()),
        "rhs_words": len(rhs.split()),
    }


def match_common_to_dyck(
    common_pool: List[Tuple[str, str]],
    dyck_pool: List[Tuple[str, str]],
    K: int,
    seed: int
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    rng = random.Random(seed)

    if len(common_pool) < K:
        raise RuntimeError(f"Common pool too small: {len(common_pool)} < {K}")
    if len(dyck_pool) < K:
        raise RuntimeError(f"Dyck pool too small: {len(dyck_pool)} < {K}")

    dyck_sample = rng.sample(dyck_pool, k=K)

    # allow replacement matching from common pool, then dedup by choosing top matches
    candidate_matches = []
    for j, (dyck_lhs, dyck_rhs) in enumerate(dyck_sample):
        fd = features(dyck_lhs, dyck_rhs)

        scored = []
        for i, (clhs, crhs) in enumerate(common_pool):
            fc = features(clhs, crhs)
            cost = (
                3 * abs(fd["lhs_chars"] - fc["lhs_chars"]) +
                4 * abs(fd["rhs_chars"] - fc["rhs_chars"]) +
                6 * abs(fd["lhs_words"] - fc["lhs_words"]) +
                8 * abs(fd["rhs_words"] - fc["rhs_words"])
            )
            scored.append((cost, i, (clhs, crhs)))

        scored.sort(key=lambda x: x[0])
        candidate_matches.append(scored)

    used_common = set()
    matched_common = [None] * K

    # greedy assignment in order of easiest-to-match first
    order = list(range(K))
    order.sort(key=lambda j: candidate_matches[j][0][0])

    for j in order:
        chosen = None
        for cost, i, pair in candidate_matches[j]:
            if i not in used_common:
                chosen = (i, pair)
                break
        if chosen is None:
            raise RuntimeError("Failed to find unique common match.")
        i, pair = chosen
        used_common.add(i)
        matched_common[j] = pair

    return matched_common, dyck_sample


def summarize_pairs(name: str, pairs: List[Tuple[str, str]]):
    lhs_chars = [len(lhs) for lhs, rhs in pairs]
    rhs_chars = [len(rhs) for lhs, rhs in pairs]
    lhs_words = [len(lhs.split()) for lhs, rhs in pairs]
    rhs_words = [len(rhs.split()) for lhs, rhs in pairs]

    print(f"\n{name} summary:")
    print(f"  count            : {len(pairs)}")
    print(f"  avg lhs chars    : {sum(lhs_chars)/len(lhs_chars):.2f}")
    print(f"  avg rhs chars    : {sum(rhs_chars)/len(rhs_chars):.2f}")
    print(f"  avg lhs words    : {sum(lhs_words)/len(lhs_words):.2f}")
    print(f"  avg rhs words    : {sum(rhs_words)/len(rhs_words):.2f}")
    print(f"  min/max lhs chars: {min(lhs_chars)} / {max(lhs_chars)}")
    print(f"  min/max rhs chars: {min(rhs_chars)} / {max(rhs_chars)}")


def format_assertions(pairs: List[Tuple[str, str]]) -> List[str]:
    return [TEMPLATE.format(lhs=lhs, rhs=rhs) for lhs, rhs in pairs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--common_out", type=str, default="common_knowledge.txt")
    ap.add_argument("--dyck_out", type=str, default="dyck_knowledge.txt")
    ap.add_argument("--K_per_group", type=int, default=200)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    common_pool = build_common_pair_pool()
    dyck_pool = build_dyck_pair_pool(args.K_per_group, args.seed)

    common_pairs, dyck_pairs = match_common_to_dyck(
        common_pool=common_pool,
        dyck_pool=dyck_pool,
        K=args.K_per_group,
        seed=args.seed,
    )

    common_lines = format_assertions(common_pairs)
    dyck_lines = format_assertions(dyck_pairs)

    Path(args.common_out).write_text("\n".join(common_lines) + "\n", encoding="utf-8")
    Path(args.dyck_out).write_text("\n".join(dyck_lines) + "\n", encoding="utf-8")

    print(f"Wrote {args.common_out} with {len(common_lines)} lines")
    print(f"Wrote {args.dyck_out} with {len(dyck_lines)} lines")

    summarize_pairs("COMMON", common_pairs)
    summarize_pairs("DYCK", dyck_pairs)

    print("\nSample COMMON examples:\n")
    for x in common_lines[:5]:
        print(x)

    print("\nSample DYCK examples:\n")
    for x in dyck_lines[:5]:
        print(x)


if __name__ == "__main__":
    main()
