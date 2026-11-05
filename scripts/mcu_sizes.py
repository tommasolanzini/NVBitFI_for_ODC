# Default distribution of multi-cell upset (MCU) sizes, for weighting results
# later. Not used for sampling: the campaigns inject each size on its own.
#
# Share of upset events by the number of cells, one event per particle. Values read
# by eye from the bar charts of the source, so about +-2 points, and normalised to
# 1: the paper prints no table. Replace them with the beam data when there are any.
#
# Source: N. J. Pieper et al., "SRAM Multi-Cell Upset Vulnerability at the 5-nm
# FinFET Node", IEEE Trans. Nuclear Science, Sandia report SAND2022-9599C,
# https://www.osti.gov/servlets/purl/2004056. Fig. 9 (terrestrial neutrons) and
# Fig. 7 (14-MeV neutrons), at the nominal 750 mV.
#
# What does not match this GPU: neutrons, not protons; test arrays of a 5-nm bulk
# FinFET, not the SRAMs of this chip; cells within 3 cells count as one event, with
# physical positions known, which software cannot see here. Lower voltage moves
# the events to larger sizes: at 550 mV, about half of them are MCUs (same paper).
#
# A check of the order of magnitude, with protons, abstract only:
# "Multiple-Cell Upset Analysis on 16/12-nm Bulk FinFET SRAM Caused by Proton
# Irradiation", https://ieeexplore.ieee.org/document/10923653/, reports at most 3
# bits on those FinFET nodes.

MCU_SIZE_SOURCE = "Pieper et al., IEEE TNS, SAND2022-9599C, Fig. 7 and 9, read from the charts"

MCU_SIZE_TABLES = {
	"terrestrial_neutrons_single_port_750mV": {1: 0.76, 2: 0.17, 3: 0.04, 4: 0.03},
	"terrestrial_neutrons_two_port_750mV": {1: 0.73, 2: 0.15, 3: 0.10, 4: 0.02},
	"14MeV_neutrons_single_port_750mV": {1: 0.76, 2: 0.23, 3: 0.01},
	"14MeV_neutrons_two_port_750mV": {1: 0.83, 2: 0.14, 3: 0.02, 4: 0.01},
}

MCU_SIZE_DEFAULT = "terrestrial_neutrons_single_port_750mV"


# Rate of a table for a set of outcome rates per size: {size: rate}. Sizes with no
# rate are left out and their share is returned apart, not assumed.
def weighted_rate(rates_by_size, table=MCU_SIZE_DEFAULT):
	shares = MCU_SIZE_TABLES[table]
	rate = sum(s * rates_by_size[k] for k, s in shares.items() if k in rates_by_size)
	return {"rate": rate, "share_without_rate": sum(s for k, s in shares.items() if k not in rates_by_size)}
