# src/const.py

CONSTANTS = {
    'G_CONV': 1.124e-5,    # Conversion factor for TOV (MeV/fm^3 -> km^-2)
    'A_CONV': 1.4766,      # Mass conversion factor (M_sun -> km)
    'P_TRANSITION': 0.08,  # Crust-Core transition pressure (MeV/fm^3)
    
    # HADRONIC CONSTRAINTS
    'H_M_MAX_UPPER': 2.35, # Maximum allowed TOV mass
    'H_M_MAX_LOWER': 1.97, # Must support J0740
    'H_R14_MIN': 11.0,     # Radius at 1.4 M_sun must be > 11km
    'H_L14_MAX': 580.0,    # GW170817 Constraint
    'H_L20_MIN': 15.0,     # Stability constraint at high mass
    
    # QUARK CONSTRAINTS
    'Q_R_MAX': 12.5,       # Maximum radius for Quark stars
    'Q_M_MIN': 0.5,        # Minimum stable mass
    'Q_CS2_RANGE': (0.33, 1.0),
    'Q_B_RANGE': (40.0, 90.0)
}