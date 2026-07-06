"""Selected parts — single source of truth for BOM, calcs, and schematic.

LCSC C-numbers verified in stock 2026-07-02/03 (JLCPCB assembly parts
library). jlc class: 'basic' (no loading fee), 'ext' (extended),
'ext-tht' (extended, wave/hand-solder THT), 'hand' (not JLC-assembled;
hand-install). Alternates recorded for order-time substitution.

Datasheet constants used by calcs.py live next to the part they belong to.
"""

PARTS = {
    # --- pulse path ---------------------------------------------------------
    "fet_pulse": dict(
        mpn="SFT040N150C3", mfr="SCILICON", lcsc="C42395334", jlc="ext",
        package="TOLL", desc="150V 4.0mOhm(max) NFET, ID(pulse) 760A",
        qty=3, alt="NCEP15T14LL (C2686615, TOLL 150V 5.8mOhm IDM 680A)",
        # calc constants
        rds_on_max_ohm=0.0040, idm_pulse_a=760.0, rth_jc_k_w=0.48,
        qg_nc=112.0, tj_max_c=150.0,
    ),
    "diode_pulse": dict(
        mpn="MBR60100DC", mfr="YFW", lcsc="C6069541", jlc="ext",
        package="TO-263", desc="100V 60A Schottky, IFSM 450A (blocking x2, freewheel x2, bank clamp x1)",
        qty=5, alt="MBR40100DC-B (C5359198, low stock)",
        vf_v=0.84, ifsm_a=450.0, i2t_a2s=450.0 ** 2 * 0.01 / 2,
    ),
    "shunt": dict(
        mpn="ASR-M-7-0.2F", mfr="Yezhan", lcsc="C469442", jlc="ext",
        package="5930", desc="0.2mOhm 1% 15W metal element (Kelvin PCB routing)",
        qty=1, alt="HolRS3920-5W-0.2mR-1% (C558625); Bourns CSS2H-3920R-L200FE (C4126121, low stock)",
        r_ohm=0.0002, p_cont_w=15.0,
    ),
    "cap_bank": dict(
        mpn="LKG1J222MESBBK", mfr="Nichicon", lcsc="C3724971", jlc="hand",
        package="snap-in 30x30mm", desc="2200uF 63V snap-in; 5 positions, populate 2 baseline (4400uF), max 11000uF",
        qty=2, qty_max=5, alt="Chengx radial 18mm 2200uF/63V (C47858)",
        c_uF=2200.0,
    ),
    "term_coil": dict(
        mpn="HB9500-9.5-2P", mfr="Kangnex", lcsc="C707836", jlc="ext-tht",  # C60785 was OOS
        package="9.5mm barrier", desc="32A barrier terminal, coil + aux-bank connections (pulse path)",
        qty=2, alt="Wurth REDCUBE M4 (hand-install, not LCSC)",
    ),
    "fuse_aux": dict(
        mpn="0997030.WXN", mfr="Littelfuse", lcsc="C207030", jlc="hand",
        package="MINI blade", desc="30A 58VDC MINI blade fuse, aux-bank feed; holder Keystone 3568",
        qty=1, alt="ATO 0287030.PXCN (C142686, 32V - marginal)",
    ),

    # --- charger -------------------------------------------------------------
    "boost_ctrl": dict(
        mpn="UC3843B", mfr="UTC", lcsc="C2995531", jlc="ext",  # C16414 not in JLC PCBA stock
        package="SOIC-8", desc="Current-mode boost controller, 24.5-55V operating (63V-rated bank) @ ~0.5A, Vcc from +12V rail",
        qty=1, alt="LM3481MMX/NOPB (C543002; needs custom symbol)",
    ),
    "boost_fet": dict(
        mpn="AOD66923", mfr="AOS", lcsc="C485687", jlc="ext",
        package="TO-252", desc="100V 11mOhm boost switch",
        qty=1, alt="UMW 15N10 (C359106)",
    ),
    "boost_diode": dict(
        mpn="SS510", mfr="MDD", lcsc="C65010", jlc="ext",
        package="SMA", desc="100V 5A Schottky, boost rectifier",
        qty=1, alt="SS510B (C7420368, SMB, preferred-class)",
    ),
    "boost_ind": dict(
        mpn="YSPI1365-470M", mfr="YJYCOIN", lcsc="C497913", jlc="ext",
        package="1265", desc="47uH 5A/7Asat shielded power inductor",
        qty=1, alt="APH1265T470M (C5349639)",
        l_uH=47.0, i_sat_a=7.0,
    ),
    "relay": dict(
        mpn="HF3FF/012-1ZS", mfr="Hongfa", lcsc="C399561", jlc="ext-tht",
        package="THT relay", desc="12V coil SPDT 15A (charge + dump), NO contact used",
        qty=2, alt="HF115F/012-1HS3 (C61377, 16A, lower stock)",
    ),
    "tl431": dict(
        mpn="TL431BIDBZR", mfr="TI", lcsc="C41283", jlc="ext",
        package="SOT-23-3", desc="Shunt ref +/-0.5%, DBZ pinout (1=K,2=REF,3=A) "
        "-- matches the TL431DBZ footprint. OVP trip 60.5V nom (59.1-62.0V band). "
        "NOT CJ431/C3113 (that part is 1=REF,2=K,3=A -> OVP dead); NOT the 1% "
        "A-grade (would blow the 62.0V ceiling)",
        qty=1, alt="CD431 +/-0.5% (JSCJ, same 1=K,2=REF,3=A pinout)",
    ),

    # --- power rails -----------------------------------------------------------
    "buck5": dict(
        mpn="AP63205WU-7", mfr="Diodes", lcsc="C2071056", jlc="ext",
        package="TSOT-23-6", desc="24V->5V/2A buck (logic + Pico VSYS + gate emitters)",
        qty=1, alt="TPS54331DR (C9865, preferred-class)",
    ),
    "reg12": dict(
        mpn="78L12(UMW)", mfr="UMW", lcsc="C347272", jlc="ext",
        package="SOT-89", desc="24V->12V linear, gate-driver + relay rail. THERMAL "
        "FOLLOW-UP: ~0.5-0.85W during charge is hot for SOT-89 -- upgrade to 78M12 "
        "TO-252/DPAK (Shikues C116325) in a dedicated re-place/re-route pass",
        qty=1, alt="78M12 TO-252 C116325 (thermal upgrade, needs re-route)",
    ),
    "tvs_in": dict(
        mpn="SMBJ26CA", mfr="MDD", lcsc="C113996", jlc="ext",
        package="SMB", desc="24V input TVS, bidirectional (matches D_TVS symbol); "
        "26V standoff clamps below SMBJ33A -- confirm vs AP63205 VIN abs-max",
        qty=1, alt="STMicro SMBJ26CA-TR (C133665)",
    ),
    "fuse_in": dict(
        mpn="0997005.WXN", mfr="Littelfuse", lcsc="C2680609", jlc="hand",
        package="MINI blade", desc="5A input fuse (holder hand-installed; confirm PN at order)",
        qty=1, alt="any 5A MINI blade",
    ),
    "pfet_rpp": dict(
        mpn="AOD4185", mfr="AOS", lcsc="C5371006", jlc="ext",
        # NOTE: was wrongly C77317 (that is a BAT54AW Schottky diode!). C5371006
        # is the AOD4185 P-channel 40V MOSFET in JLC assembly stock (TO-252).
        package="TO-252", desc="-40V PFET reverse-polarity protection (confirm stock at order)",
        qty=1, alt="any 40V+ PFET, TO-252",
    ),
    "jack24": dict(
        mpn="DC-005-5A-2.0", mfr="XKB", lcsc="C381116", jlc="ext-tht",
        package="DC barrel 5.5/2.1", desc="24V input jack (5A, 24V rated - input only)",
        qty=1, alt="5.08mm terminal WJ128V-5.0-2P (C8269)",
    ),

    # --- switch drive + interlock ----------------------------------------------
    "gate_drv": dict(
        mpn="UCC27524ADR", mfr="TI", lcsc="C185857", jlc="ext",
        package="SOIC-8", desc="Dual 5A low-side gate driver (per-FET gate resistors)",
        qty=2, alt="UCC27524DR (C465729)",  # 2 pkgs: 3 FETs + spare channel
    ),
    "and_gate": dict(
        mpn="SN74LVC1G08DCKR", mfr="TI", lcsc="C7832", jlc="ext",
        package="SC-70-5", desc="FIRE = MCU_FIRE AND ARM (hardware interlock)",
        qty=1, alt="SN74LVC1G08DBVR (C7666, SOT-23-5)",
    ),
    "gate_clamp": dict(
        mpn="BZT52C15", mfr="-", lcsc="C19077412", jlc="ext",
        package="SOD-123", desc="15V gate clamp zener per FET",
        qty=3, alt="SMAJ15A (C19077535)",
    ),

    # --- sense -------------------------------------------------------------------
    "ina240": dict(
        mpn="INA240A1PWR", mfr="TI", lcsc="C93965", jlc="ext",
        package="TSSOP-8", desc="Current-sense amp, gain 20 (0.2mOhm -> 4mV/A)",
        qty=1, alt="INA240A2PWR (C129949, gain 50 - would clip)",
        gain=20.0,
    ),
    "adc": dict(
        mpn="ADS7042IDCUR", mfr="TI", lcsc="C701641", jlc="ext",
        package="VSSOP-8", desc="12-bit 1MSPS SPI ADC for I(t) capture",
        qty=1, alt="ADS7886SBDBVR (C2669886, SOT-23-6)",
    ),
    "opamp": dict(
        mpn="MCP6002T-I/SN", mfr="Microchip", lcsc="C7377", jlc="ext",
        package="SOIC-8", desc="RRIO dual op-amp: V_bank + 24V sense buffers",
        qty=1, alt="LMV358IDR (C63813)",
    ),
    "comparator": dict(
        mpn="LM339DR", mfr="TI", lcsc="C7948", jlc="ext",
        package="SOIC-14", desc="Quad comparator x2: 6 IR gate channels + live-bank detect",
        qty=2, alt="LM339DR2G (C63821)",
    ),

    # --- gates / connectors -------------------------------------------------------
    "ir_led": dict(
        mpn="IR333C-A", mfr="Everlight", lcsc="C5130", jlc="ext-tht",
        package="5mm THT", desc="940nm IR emitter (gate-rail EMIT variant)",
        qty=6, alt="IR204C-A-L (C98863, 3mm)",
    ),
    "ir_pt": dict(
        mpn="PT204-6B", mfr="Everlight", lcsc="C5133", jlc="ext-tht",
        package="3mm THT", desc="940nm NPN phototransistor (gate-rail RECV variant)",
        qty=6, alt="PT333-3C (C264295, OUT OF STOCK 2026-07)",
    ),
    "idc10": dict(
        mpn="2.54-2*5P box header", mfr="BOOMELE", lcsc="C5665", jlc="ext-tht",
        package="IDC 2x5", desc="Sensor-rail ribbon headers (2 on driver, 1 per rail)",
        qty=4, alt="any shrouded 2x5 2.54mm",
    ),
    "jst_xh4": dict(
        mpn="B4B-XH-A(LF)(SN)", mfr="JST", lcsc="C144395", jlc="ext-tht",
        package="JST-XH 4p", desc="Spare ad-hoc sensor connector",
        qty=1, alt="B4B-XH-AM (C161871)",
    ),

    "r_precharge": dict(
        mpn="SQM5W-47R-5%", mfr="VO", lcsc="C5807995", jlc="ext-tht",
        package="cement THT", desc="47R 5W precharge across charge-relay contacts",
        qty=1, alt="TyOhm SMW 5W 47R (C2937346, SMD)",
    ),
    "r_dump": dict(
        mpn="CR-M10W100R", mfr="CCO", lcsc="C5110275", jlc="ext-tht",  # C216413 was OOS
        package="cement THT", desc="100R 10W x2 in series = 200R/20W dump (5.3s to <5V)",
        qty=2, alt="RX27-4V 20W from Digi-Key if single-part preferred",
    ),
    "r_bleed": dict(
        mpn="25121WJ0682T4E", mfr="UNI-ROYAL", lcsc="C26073", jlc="ext",
        package="2512 1W", desc="6.8k permanent bleed (0.44W cont, 2.2x margin). "
        "1W is the 2512 land rating; 1.5W/2W 2512 are all 0-stock on LCSC",
        qty=1, alt="C340514 (RMC1-682JTE-M) or C42372225, both 6.8k 2512 1W in stock",
    ),
    "r_boost_cs": dict(
        mpn="HoJLR2512-3W-150mR-1%", mfr="Milliohm", lcsc="C2903485", jlc="ext",
        package="2512 3W", desc="0.15ohm boost switch current sense (~6.7A limit; "
        "was 0.05ohm/20A, above realistic inductor saturation)",
        qty=1, alt="TA-I RLP25FEER150 (C459673)",
    ),
    "cap_boost_out": dict(
        mpn="RVT1J101M1010", mfr="Honor", lcsc="C28241", jlc="ext",
        package="SMD 10x10.2", desc="100uF 63V boost output x2 (ripple: datasheet check at order)",
        qty=2, alt="Lelon VE-101M1JTR-1010 (C249468)",
    ),
    "cap_bulk_35v": dict(
        mpn="VEJ101M1VTR-0607", mfr="Lelon", lcsc="C176666", jlc="ext",
        package="SMD 6.3x7.7", desc="100uF 35V input bulk + armed-rail reservoir",
        qty=2, alt="",
    ),
    "ind_buck": dict(
        mpn="SWPA6045S100MT", mfr="Sunlord", lcsc="C79272", jlc="ext",
        package="6x6x4.5", desc="10uH Isat 3.5A, 5V buck output",
        qty=1, alt="SWPA5040S100MT (C84608)",
    ),
    "buzzer": dict(
        mpn="YS-MBZ12085C05R42", mfr="Fengming", lcsc="C409842", jlc="ext-tht",
        package="12mm THT", desc="5V passive electromagnetic buzzer",
        qty=1, alt="MLT-8530 SMD (C94599, 2.5-4.5V)",
    ),
    "fuse_holder": dict(
        mpn="Keystone 3568", mfr="Keystone", lcsc="C5249699", jlc="ext-tht",
        package="MINI clip THT", desc="MINI blade fuse holder (2 clips per fuse position)",
        qty=4, alt="",
    ),

    # --- hand-install (not JLC) -----------------------------------------------------
    "pico_socket": dict(
        mpn="2.54-1*20P female", mfr="BOOMELE", lcsc="C50984", jlc="ext-tht",
        package="THT", desc="Raspberry Pi Pico socket (Pico bought separately)",
        qty=2, alt="",
    ),
    "io_breakout": dict(
        mpn="B-2100S40P-B110 2x20 breakaway", mfr="Ckmtw", lcsc="C124383", jlc="ext-tht",
        package="THT", desc="MCU-agnostic IO breakout (parallel with Pico socket)",
        qty=1, alt="",
    ),
    "arm_key": dict(
        mpn="key switch SPST", mfr="-", lcsc="", jlc="hand",
        package="panel", desc="ARM keyswitch (panel-mount, wired to header)",
        qty=1, alt="",
    ),
}

# Bank options achievable with the 5 snap-in positions (2200uF each)
BANK_POSITIONS = 5
BANK_UNIT_UF = 2200.0

# Demo-coil PHYSICAL BUILD SPEC (experiment #1). Single source of truth for
# hardware/README.md's build sheet and tests/test_coil_physics.py's
# field-geometry contract test. The field model (analytical_bfield.
# solenoid_field) places N ideal loops at loop_center_radius_mm with
# centers np.linspace(-span/2, +span/2, N) -- the physical winding must
# match LOOP CENTERS, not wire edges.
DEMO_COIL_BUILD = {
    "n_turns": 30,
    "loop_center_radius_mm": 15.0,     # = (inner 12 + outer 18)/2 of config
    "center_span_mm": 30.0,            # first-to-last turn CENTER distance
    "pitch_mm": 30.0 / 29,             # ~1.034 center-to-center (span/(N-1))
    "wire_copper_mm": 0.8,             # AWG20 conductor
    "wire_finished_od_mm": 0.87,       # copper + 2 x 0.035 enamel
    "former_od_mm": 2 * (15.0 - 0.87 / 2),  # ~29.13; set CENTER radius, not OD
    "expected_L_uH": 17.3,             # Wheeler at 15mm (config says 12.4 -- see
    "expected_R_dc_mohm": 97,          #  model-inconsistency note); measure and
    "expected_R_total_mohm": 127,      #  write actuals into coil_params.json
}
