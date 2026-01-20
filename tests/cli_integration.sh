#!/bin/bash
# CLI integration tests - compare output with expected values from C++ voronota-lt

set -e

BINARY="${1:-./target/release/voronotalt}"
DATA_DIR="benches/data"

# Helper to check approximate equality
check_approx() {
    local name="$1"
    local actual="$2"
    local expected="$3"
    local tolerance="$4"

    local diff=$(echo "$actual - $expected" | bc -l)
    local abs_diff=$(echo "if ($diff < 0) -1*$diff else $diff" | bc -l)

    if (( $(echo "$abs_diff > $tolerance" | bc -l) )); then
        echo "FAIL: $name - expected $expected, got $actual (diff: $abs_diff > $tolerance)"
        return 1
    else
        echo "PASS: $name = $actual (expected $expected, tolerance $tolerance)"
        return 0
    fi
}

# Helper to extract value from summary output
get_value() {
    echo "$1" | grep "^$2:" | cut -d: -f2 | tr -d ' '
}

echo "=== Testing balls_cs_1x1.xyzr with probe=2.0 ==="

OUTPUT=$($BINARY -i "$DATA_DIR/balls_cs_1x1.xyzr" --probe 2.0 -q)

check_approx "contacts" "$(get_value "$OUTPUT" "contacts")" "153" "0"
check_approx "cells" "$(get_value "$OUTPUT" "cells")" "100" "0"
check_approx "total_contact_area" "$(get_value "$OUTPUT" "total_contact_area")" "3992.55" "1"
check_approx "total_sas_area" "$(get_value "$OUTPUT" "total_sas_area")" "21979.6" "10"
check_approx "total_volume" "$(get_value "$OUTPUT" "total_volume")" "46419.9" "10"

echo ""
echo "=== Testing balls_cs_1x1.xyzr periodic with probe=2.0 ==="

OUTPUT=$($BINARY -i "$DATA_DIR/balls_cs_1x1.xyzr" --probe 2.0 --periodic-box-corners 0 0 0 200 250 300 -q)

check_approx "contacts" "$(get_value "$OUTPUT" "contacts")" "189" "0"
check_approx "cells" "$(get_value "$OUTPUT" "cells")" "100" "0"
check_approx "total_contact_area" "$(get_value "$OUTPUT" "total_contact_area")" "4812.14" "50"
check_approx "total_sas_area" "$(get_value "$OUTPUT" "total_sas_area")" "20023.1" "100"
check_approx "total_volume" "$(get_value "$OUTPUT" "total_volume")" "45173.2" "100"

echo ""
echo "=== Testing balls_2zsk.xyzr with probe=1.4 ==="

OUTPUT=$($BINARY -i "$DATA_DIR/balls_2zsk.xyzr" --probe 1.4 -q)

# C++ expected: 23855 contacts, 3545 cells
check_approx "contacts" "$(get_value "$OUTPUT" "contacts")" "23855" "0"
check_approx "cells" "$(get_value "$OUTPUT" "cells")" "3545" "0"

echo ""
echo "=== Testing edge case: 3 balls in a line ==="

OUTPUT=$(echo -e "0 0 0 1\n0.5 0 0 1\n1 0 0 1" | $BINARY --probe 1.0 -q)
CONTACTS=$(get_value "$OUTPUT" "contacts")
CELLS=$(get_value "$OUTPUT" "cells")

if [ "$CELLS" -eq 3 ]; then
    echo "PASS: 3 balls in line produces 3 cells"
else
    echo "FAIL: 3 balls in line - expected 3 cells, got $CELLS"
fi

echo ""
echo "=== Testing print-contacts output format ==="

CONTACT_OUTPUT=$($BINARY -i "$DATA_DIR/balls_cs_1x1.xyzr" --probe 2.0 --print-contacts -q | head -1)
# Expected format: "id_a id_b area arc_length"
# First contact should be: 0 1 42.9555 23.2335

ID_A=$(echo "$CONTACT_OUTPUT" | awk '{print $1}')
ID_B=$(echo "$CONTACT_OUTPUT" | awk '{print $2}')
AREA=$(echo "$CONTACT_OUTPUT" | awk '{print $3}')

if [ "$ID_A" = "0" ] && [ "$ID_B" = "1" ]; then
    echo "PASS: First contact IDs correct (0, 1)"
else
    echo "FAIL: First contact IDs - expected 0 1, got $ID_A $ID_B"
fi

check_approx "first_contact_area" "$AREA" "42.9555" "0.01"

echo ""
echo "=== All tests completed ==="
