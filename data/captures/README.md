# Scope captures (Siglent SDS1202X-E via host/scope.py, omnimarble-vbench)

Columns: `t_s, i_A, v_bank_V`. Trigger on the bank edge (CH2 falling),
centred, 7 ms window, boxcar-averaged to ~2800 rows. `i_A` is the J6
shunt current (0.2 mV/A) -- the shunt is in the FET source leg and NEVER
sees the freewheel tail (TWIN_AUDIT S-9); integrals are shunt-visible only.

| file | date | bank | notes |
|---|---|---|---|
| scope_3can_10v_700.csv | 2026-08-29 | 9.65 V | blank |
| scope_3can_49v_{400,700,1500}.csv | 2026-08-29 | 49.15 V | blanks; tau=275 us tail fitted from these + marble dv |
| scope_3can_30v_700.csv | 2026-09-09 | 29.45 V | blank; 30 V injected prediction 69.6 -> measured 73.3 |
| scope_4can_49v_{400,700,1000,1500}.csv | 2026-09-24 | 49.8 V | blanks, 100 ohm charge resistor |
| scope_4can_22ohm_blank_49v_700.csv | 2026-09-24 | 49.77 V | blank after the 22 ohm swap (inert to the pulse) |
| scope_4can_marble_plus9_49v_700.csv | 2026-09-24 | 49.30 V | FIRST marble-shot capture, +9 offset; early slope -4.4% vs blank |
| scope_5can_49v_{400,700,1000,1500}.csv | 2026-09-24 | 49.8 V | blanks, 22 ohm resistor, PSU 1.0 A |

Scoring and injected predictions: `docs/PREDICTION_45CAN.md`.
