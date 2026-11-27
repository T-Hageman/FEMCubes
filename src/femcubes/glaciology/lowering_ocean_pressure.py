from ufl import max_value, min_value

from femcubes.glaciology.ocean_pressure import OceanPressure


class LoweringOceanPressure(OceanPressure):
    r"""Hydrostatic ocean pressure with a water level lowered at a constant rate.

    The model is :doc:`ocean_pressure` with a sea level that changes in time:
    the water stands at ``WaterHeight`` up to ``LoweringStart``, is then
    lowered at a constant rate, and stays at ``FinalWaterHeight`` from
    ``LoweringStart + LoweringTime`` on. Lowering the water level gradually
    removes the hydrostatic support of an ice front, which loads an ice cliff
    progressively, as when the buttressing water in front of it drains.
    Fields, Jacobian and the displacement or velocity formulation are those
    of :doc:`ocean_pressure`.

    .. rubric:: Weak form

    As :doc:`ocean_pressure`, with the constant :math:`h_w` replaced by

    .. math::

        h_w(t) = h_0 - (h_0 - h_1)\,
        \min\Bigl(\max\Bigl(\frac{t - t_0}{T}, 0\Bigr), 1\Bigr),

    with :math:`h_0` = ``WaterHeight``, :math:`h_1` = ``FinalWaterHeight``,
    :math:`t_0` = ``LoweringStart`` and :math:`T` = ``LoweringTime``,
    evaluated at the end of the time step, :math:`t = t_n + \Delta t`.

    Example model settings
    ----------------------

    .. code-block:: python

        class Params(ParamsTemplate):
            dim = 2                          # REQUIRED: 2 or 3
            gravity = -9.81                  # REQUIRED: acceleration along the last axis (m/s^2)
            Solution_Steps = {"u": 0}        # REQUIRED: "u" (or "v"); "phasefield" with DamagedForces
            Materials = {
                "Water": {                   # REQUIRED: always read from Materials["Water"]
                    "Density": 1025.0,       # REQUIRED: water density rho_w (kg/m^3)
                    "WaterHeight": 177.6,    # OPTIONAL (default 0.0): sea level h_0 before the lowering (m)
                },
            }
            Models = {
                "Ocean": {
                    "Type": "glaciology:LoweringOceanPressure",  # REQUIRED
                    "boundary": "right",                 # REQUIRED: boundary group in contact with water
                    "LoweringTime": 3600.0,              # REQUIRED: duration T of the lowering (s), > 0
                    "FinalWaterHeight": 0.0,             # OPTIONAL (default 0.0): sea level h_1 after the lowering (m)
                    "LoweringStart": 0.0,                # OPTIONAL (default 0.0): time t_0 the lowering starts (s)
                    "DamagedForces": False,              # OPTIONAL (default False): degrade the pressure with phasefield
                    "domain": "domain",                  # OPTIONAL (default "domain"): domain of the ice (and of v)
                },
            }
    """

    MODEL_TYPE = "glaciology:LoweringOceanPressure"

    def __init__(self, name, params, mesh):
        super().__init__(name, params, mesh)
        settings = self.params.Models[self.name]
        self.hw_end = float(settings.get("FinalWaterHeight", 0.0))
        self.lowering_start = float(settings.get("LoweringStart", 0.0))
        if "LoweringTime" not in settings:
            raise ValueError(f"{self.MODEL_TYPE} '{name}': LoweringTime is required.")
        self.lowering_time = float(settings["LoweringTime"])
        if self.lowering_time <= 0.0:
            raise ValueError(f"{self.MODEL_TYPE} '{name}': LoweringTime must be positive.")

    def water_height(self, t: float) -> float:
        """Sea level at time ``t`` (the law of the weak form, for post-processing)."""
        fraction = min(max((t - self.lowering_start) / self.lowering_time, 0.0), 1.0)
        return self.hw - (self.hw - self.hw_end) * fraction

    def _water_height(self, time_constant, dt_constant):
        # Symbolic in the time constants, so the compiled form follows the water level.
        fraction = (time_constant + dt_constant - self.lowering_start) / self.lowering_time
        fraction = min_value(max_value(fraction, 0.0), 1.0)
        return self.hw - (self.hw - self.hw_end) * fraction
