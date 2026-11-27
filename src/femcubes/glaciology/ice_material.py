"""Ice material properties as functions of porosity and temperature."""

import numpy as np
import ufl

from femcubes.core.utils.maths_utils import ft_to_Gc


class IceMaterial:
    r"""Ice-specific material properties for glacier and ice-shelf models.

    Helper class, not a model: the glaciology models evaluate their material
    parameters from the porosity :math:`p` (``poros``) and the temperature
    :math:`T` (``T``, in degrees Celsius), so that the properties can vary
    through an ice body, for example from porous firn near the surface to
    solid ice at depth. Porosity reduces the density and the stiffness and
    increases the creep rate, while temperature sets the rate factor of
    Glen's flow law :cite:p:`Glen1955,Paterson1982`. All methods accept
    numbers, NumPy arrays or UFL expressions; they do not check their inputs
    (the ice models check :math:`0\leq p<1` and :math:`T\leq0` °C on their
    fields, see :attr:`POROSITY_RANGE` and :attr:`MELTING_TEMPERATURE`). The
    relations are

    * density :math:`\rho = 917\,(1-p)` kg/m^3 (:meth:`Get_Density`);
    * Young's modulus :math:`E = 9.5\times10^{9}\,(1-p)^4` Pa and Poisson's
      ratio :math:`\nu = 0.35`, giving :math:`\lambda`, :math:`\mu` and
      :math:`K` by the isotropic relations (:meth:`Get_Lame_Parameter`,
      :meth:`Get_Shear_Modulus`, :meth:`Get_Bulk_Modulus`);
    * tensile strength :math:`f_t = 0.2` MPa, independent of :math:`T` and
      :math:`p` (:meth:`Get_Tensile_Strength`, which gives the sources);
    * Glen exponent :math:`n = 3` (:meth:`Get_Creep_n`);
    * Glen rate factor (Pa^-n s^-1, :meth:`Get_Creep_A`)
      :math:`A = A_0\exp\bigl(-\tfrac{Q}{R}(\tfrac{1}{T_K}-\tfrac{1}{263.15})\bigr)\exp\bigl(\tfrac{4p}{1-p}\bigr)`
      with :math:`T_K = T + 273.15`, :math:`Q = 60` kJ/mol below
      :math:`T_K = 263.15` K and :math:`Q = 139` kJ/mol above it, and
      :math:`A_0 = 3.615\times10^{-13}\exp(-60\times10^{3}/(263.15\,R))`,
      the Arrhenius constants of :cite:t:`Paterson1982`;
    * return-mapping stress scale
      :math:`\eta_\mathrm{vp} = \bigl(d^{(n+1)/2}/(2A)\bigr)^{1/n}`, with
      :math:`d = 3`, or :math:`d = 2` for intrinsic 2-D models
      (:meth:`Get_ViscoPlastic_viscosity`), and shallow-ice viscosity
      factor :math:`\tfrac12 A^{-1/n}` (:meth:`Get_Viscosity_Stokes`);
    * phase-field toughness (:meth:`Get_Gc`): :math:`G_c = 10\,(1-p)` J/m^2
      for ``AT2``, independent of :math:`T`; for ``AT1``, ``HO-AT1`` and
      ``HO-AT2`` the strength calibration
      :func:`~femcubes.core.utils.maths_utils.ft_to_Gc`,
      :math:`G_c = c\,f_t^2\ell/E(p)` (:math:`c = 8/3` for the AT1 types,
      :math:`256/27` for ``HO-AT2``), which grows with porosity as
      :math:`(1-p)^{-4}` and, with the constant :math:`f_t`, does not depend
      on :math:`T`.
    """
    
    # Universal constants
    R = 8.31446261815324 # J/(mol*K), universal gas constant
    
    n = 3.0 # stress exponent for Glen's flow law (used across multiple functions)

    #: Admissible porosity, ``POROSITY_RANGE[0] <= p < POROSITY_RANGE[1]``.
    POROSITY_RANGE = (0.0, 1.0)
    #: Melting temperature (°C) of the relations; ice temperatures must not exceed it.
    MELTING_TEMPERATURE = 0.0
    #: Tensile strength (Pa) returned by :meth:`Get_Tensile_Strength`.
    TENSILE_STRENGTH = 0.2e6
    
    ### Mechanical properties
    def Get_Density(self, porosity, temperature):
        """Get density from dry porosity, neglecting thermal expansion.

        References
        ----------
        The solid-ice value ``rho_0 = 917 kg m^-3`` is consistent with
        :cite:t:`Rees1992` and :cite:t:`Clayton2024a`. The dry porosity relation
        ``rho = rho_i * (1 - porosity)`` is used by :cite:t:`Sidler2015`.
        """
        rho_0 = 917.0 # kg/m^3, density of solid ice at 0C
        
        return rho_0 * (1 - porosity)
    
    def Get_Youngs_Modulus(self, porosity, temperature):
        """Get Young's modulus using a local relative-density power law.

        References
        ----------
        The solid-ice value ``E0 = 9.5 GPa`` is consistent with
        :cite:t:`Gammon1983`, :cite:t:`Sidler2015` and
        :cite:t:`Clayton2024a`. The ``(1 - porosity)^4`` scaling is an
        empirical local model of the weakening of porous ice.
        """
        E0 = 9.5e9 # Pa, Young's modulus of solid ice at 0C
        return E0 * (1 - porosity)**4
    
    def Get_Poisson(self, porosity, temperature):
        """Get Poisson's ratio, assumed porosity and temperature independent.

        References
        ----------
        The value ``poisson = 0.35`` is consistent with :cite:t:`Sidler2015`
        and :cite:t:`Clayton2024a`.
        """
        poisson = 0.35 # Poisson's ratio of solid ice at 0C
        return poisson 
    
    def Get_Bulk_Modulus(self, porosity, temperature):
        """Get bulk modulus from isotropic linear elasticity.

        References
        ----------
        The conversion from Young's modulus and Poisson's ratio follows
        standard isotropic linear elasticity relations
        """
        E = self.Get_Youngs_Modulus(porosity, temperature)
        nu = self.Get_Poisson(porosity, temperature)
        return E / (3 * (1 - 2*nu))
    
    def Get_Shear_Modulus(self, porosity, temperature):
        """Get shear modulus from isotropic linear elasticity.

        References
        ----------
        The conversion from Young's modulus and Poisson's ratio follows
        standard isotropic linear elasticity relations.
        """
        E = self.Get_Youngs_Modulus(porosity, temperature)
        nu = self.Get_Poisson(porosity, temperature)
        return E / (2 * (1 + nu))
    
    def Get_Lame_Parameter(self, porosity, temperature):
        """Get the first Lame parameter from isotropic linear elasticity.

        References
        ----------
        The conversion from Young's modulus and Poisson's ratio follows
        standard isotropic linear elasticity relations.
        """
        E = self.Get_Youngs_Modulus(porosity, temperature)
        nu = self.Get_Poisson(porosity, temperature)
        lambda_ = E * nu / ((1 + nu) * (1 - 2*nu))
        return lambda_
    
    def Get_Tensile_Strength(self, porosity, temperature):
        """Return the tensile strength :math:`f_t` (Pa), the constant :attr:`TENSILE_STRENGTH`.

        A number for number or UFL inputs, an array of the broadcast shape of
        array inputs.

        References
        ----------
        The data do not support a temperature law for the tensile strength
        of polycrystalline ice, so :math:`f_t` does not depend on :math:`T`.
        In uniaxial tension tests on polycrystalline (snow-)ice from
        -0.1 °C to -54 °C, :cite:t:`Haynes1978` found the strength
        relatively insensitive to temperature, between 1.7 and 3.2 MPa
        (about 20 % higher at -18 °C than at -0.1 °C), and the review of
        :cite:t:`Petrovic2003` gives 0.7 to 3.1 MPa, relatively insensitive
        to temperature and strain rate and decreasing with grain size. At the
        scale of glaciers, :cite:t:`Vaughan1993` inferred tensile strengths
        of 90 to 320 kPa from the occurrence of crevasses at 17 polar and
        alpine sites, with no systematic variation with the ice temperature
        at 10 m depth. The value 0.2 MPa lies in this field range, which
        applies to near-surface ice and firn at glacier scale; laboratory
        specimens are an order of magnitude stronger. :math:`f_t` is also
        taken independent of the porosity.
        """
        if self._Uses_UFL(porosity, temperature):
            return self.TENSILE_STRENGTH
        shape = np.broadcast(np.asarray(porosity), np.asarray(temperature)).shape
        return self.TENSILE_STRENGTH if shape == () else np.full(shape, self.TENSILE_STRENGTH)
    
    def Get_Gc(self, porosity, temperature, ell0, PF_formulation="AT2"):
        """Get the phase-field fracture toughness (J/m^2) of the local ice state.

        Parameters
        ----------
        porosity:
            Local ice porosity.
        temperature:
            Local ice temperature in degrees Celsius.
        ell0:
            Phase-field regularization length scale in metres.
        PF_formulation:
            Phase-field formulation used to calibrate fracture nucleation.
            Supported values are ``"AT1"``, ``"AT2"``, ``"HO-AT1"``, and
            ``"HO-AT2"``.

        Notes
        -----
        ``AT2`` prescribes ``Gc = 10*(1-porosity)`` J/m^2, independent of
        the temperature and of ``ell0``, for use with the cohesive models
        (whose strength does not follow from ``Gc``). The other formulations
        use the strength calibration
        :func:`~femcubes.core.utils.maths_utils.ft_to_Gc`,
        ``Gc = c*f_t**2*ell0/E`` with ``c = 8/3`` (``AT1``, ``HO-AT1``) or
        ``256/27`` (``HO-AT2``, the AT2 peak-stress calibration): with the
        constant :meth:`Get_Tensile_Strength` and ``E ~ (1-porosity)**4``,
        this toughness grows with porosity as ``(1-porosity)**-4`` and does
        not depend on the temperature.
        """
        if PF_formulation == "AT2": # Temporary: Assuming if AT2 is picked, then cohesive phase-field will be used
            Gc = 10.0*(1.0-porosity)
            return Gc
        else:
            f_t = self.Get_Tensile_Strength(porosity, temperature)
            E = self.Get_Youngs_Modulus(porosity, temperature)
            return ft_to_Gc(f_t, E, ell0, PF_formulation=PF_formulation)

    ### viscous properties
    def Get_Creep_n(self, porosity, temperature):
        """Get Glen's flow-law exponent ``n``."""
        return self.n
    
    def Get_Creep_A(self, porosity, temperature):
        """Get Glen's flow-law rate factor ``A``.

        The returned value has units Pa^-n s^-1. Temperature dependence uses
        the cold branch below -10 C and the warm/temperate branch above it.
        Porosity increases the rate factor through the existing exponential
        softening term.

        References
        ----------
        Glen's power-law creep follows :cite:t:`Glen1955`. The split
        Arrhenius constants follow :cite:t:`Paterson1982`, and are also
        documented in PISM:
        https://www.pism.io/docs/manual/modeling-choices/dynamics/rheology.html.
        """
        
        T_glen_transition = 263.15 # K, transition between cold and warm branches
        A_glen_cold = 3.615e-13 # Pa^-3 s^-1, cold-branch pre-exponential factor
        Q_glen_cold = 60e3 # J/mol, cold-branch activation energy
        Q_glen_temperate = 139e3 # J/mol, warm/temperate activation energy
        A_glen_transition = A_glen_cold * np.exp(-Q_glen_cold / (self.R * T_glen_transition))
        porosity_glen_beta = 4.0 # factor controlling porosity dependence of Glen's flow law, with A ~ exp(beta * p/(1-p)))
        
        use_ufl = self._Uses_UFL(porosity, temperature)
        T = temperature + 273.15 # convert to Kelvin
        A_cold = A_glen_transition * self._Exp( -Q_glen_cold / self.R * (1.0 / T - 1.0 / T_glen_transition), use_ufl)
        A_temperate = A_glen_transition * self._Exp(-Q_glen_temperate / self.R * (1.0 / T - 1.0 / T_glen_transition),use_ufl)
        A = self._Where_Less_Than(T,T_glen_transition,A_cold,A_temperate,use_ufl)
        A = A * self._Exp(porosity_glen_beta * porosity / (1 - porosity),use_ufl) # add porosity dependence
        return A

    def Get_Viscosity_Stokes(self,porosity,temperature):
        """Get the base viscosity factor used by Glen's law Stokes flow.

        References
        ----------
        The viscosity-form convention for Glen-type Stokes flow is used by
        :cite:t:`Bueler2009`, and documented in PISM:
        https://www.pism.io/docs/manual/modeling-choices/dynamics/rheology.html.
        """

        A = self.Get_Creep_A(porosity, temperature)
        eta = 0.5 * A ** (-1.0 / self.n)

        return eta
    
    def Get_ViscoPlastic_viscosity(self, porosity, temperature, dim=3):
        """Get the J2 stress scale used by return-mapping Glen creep.

        The returned value :math:`\\eta_\\mathrm{vp} = (d^{(n+1)/2}/(2A))^{1/n}`
        is the stress scale of the creep multiplier rate
        :math:`\\dot\\Lambda_v = (f/\\eta_\\mathrm{vp})^n` of the return mapping, with
        :math:`f = \\sqrt{(d/2)\\,\\mathbf{s}:\\mathbf{s}}`, so that the tensor creep rate is Glen's
        :math:`A\\tau_e^{n-1}\\mathbf{s}`. ``dim`` is the dimension
        :math:`d` of the constitutive law: 3 (plane strain, plane stress and
        3-D) or 2 (intrinsic 2-D, ``true_2d``), as in
        :doc:`../solids/viscoelastic`.

        References
        ----------
        Glen's power-law creep follows :cite:t:`Glen1955`. The split
        Arrhenius constants follow :cite:t:`Paterson1982`, and are also
        documented in PISM:
        https://www.pism.io/docs/manual/modeling-choices/dynamics/rheology.html.
        """
        if dim not in (2, 3):
            raise ValueError(f"dim must be 2 or 3, got {dim!r}.")
        A = self.Get_Creep_A(porosity, temperature)
        return (float(dim) ** ((self.n + 1.0) / 2.0) / (2.0 * A)) ** (1.0 / self.n)

    def Plot_Relations(
        self,
        ToPlot=None,
        temperature_range=(-50.0, 0.0),
        porosity_range=(0.0, 0.8),
        num_temperature=200,
        num_porosity=200,
        show=True,
        block=False,
        draw_pause=0.1,
    ):
        """Plot material relations over temperature and porosity.

        Parameters
        ----------
        ToPlot:
            Relation name or iterable of relation names. Names may be supplied
            with or without the ``Get_`` prefix, for example ``"Density"`` or
            ``"Get_Density"``. If omitted, all available relations are plotted.
        temperature_range:
            Two-value Celsius range for the horizontal axis.
        porosity_range:
            Two-value porosity range for the vertical axis. Values must satisfy
            ``0 <= porosity < 1``.
        num_temperature:
            Number of temperature samples.
        num_porosity:
            Number of porosity samples.
        show:
            If ``True``, call ``matplotlib.pyplot.show`` before returning.
        block:
            Passed to ``matplotlib.pyplot.show`` when ``show`` is ``True``.
            Defaults to ``False`` so plots do not stop subsequent simulation
            steps from running.
        draw_pause:
            Seconds to briefly process GUI events after a non-blocking show.
            This gives interactive backends time to render the window.

        Returns
        -------
        tuple
            ``(fig, axes)``. ``axes`` is a single axis when one relation is
            requested, otherwise it is a one-dimensional NumPy array of axes.
        """

        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm

        relations = {
            "Density": (self.Get_Density, "Density [kg m$^{-3}$]"),
            "Youngs_Modulus": (self.Get_Youngs_Modulus, "Young's modulus [Pa]"),
            "Poisson": (self.Get_Poisson, "Poisson's ratio [-]"),
            "Bulk_Modulus": (self.Get_Bulk_Modulus, "Bulk modulus [Pa]"),
            "Shear_Modulus": (self.Get_Shear_Modulus, "Shear modulus [Pa]"),
            "Lame_Parameter": (self.Get_Lame_Parameter, "Lame parameter [Pa]"),
            "Creep_A": (self.Get_Creep_A, "Glen rate factor [Pa$^{-n}$ s$^{-1}$]"),
            "Tensile_Strength": (self.Get_Tensile_Strength, "Tensile strength [Pa]"),
            "Viscosity_Stokes": (
                self.Get_Viscosity_Stokes,
                "Glen base viscosity coefficient [Pa s$^{1/n}$]",
            ),
            "ViscoPlastic_viscosity": (
                self.Get_ViscoPlastic_viscosity,
                "Viscoplastic reference stress [Pa s$^{1/n}$]",
            ),
        }
        aliases = self._Get_Plot_Relation_Aliases(relations)

        if ToPlot is None:
            relation_names = list(relations)
        elif isinstance(ToPlot, str):
            relation_names = [ToPlot]
        else:
            relation_names = list(ToPlot)

        if not relation_names:
            raise ValueError("ToPlot must contain at least one relation name.")

        selected_names = []
        for relation_name in relation_names:
            relation_key = self._Normalise_Plot_Relation_Name(relation_name)
            if relation_key not in aliases:
                available = ", ".join(relations)
                raise ValueError(
                    f"Unknown material relation '{relation_name}'. "
                    f"Available relations: {available}."
                )
            selected_names.append(aliases[relation_key])

        temperature_range = np.asarray(temperature_range, dtype=float)
        porosity_range = np.asarray(porosity_range, dtype=float)
        num_temperature = int(num_temperature)
        num_porosity = int(num_porosity)

        if temperature_range.shape != (2,):
            raise ValueError("temperature_range must contain exactly two values.")
        if porosity_range.shape != (2,):
            raise ValueError("porosity_range must contain exactly two values.")
        if temperature_range[0] >= temperature_range[1]:
            raise ValueError("temperature_range must be strictly increasing.")
        if temperature_range[0] <= -273.15:
            raise ValueError("temperature_range must be above absolute zero.")
        if porosity_range[0] < 0.0 or porosity_range[1] >= 1.0:
            raise ValueError("porosity_range must satisfy 0 <= porosity < 1.")
        if porosity_range[0] >= porosity_range[1]:
            raise ValueError("porosity_range must be strictly increasing.")
        if num_temperature < 2:
            raise ValueError("num_temperature must be at least 2.")
        if num_porosity < 2:
            raise ValueError("num_porosity must be at least 2.")

        temperature = np.linspace(
            temperature_range[0],
            temperature_range[1],
            num_temperature,
        )
        porosity = np.linspace(
            porosity_range[0],
            porosity_range[1],
            num_porosity,
        )
        temperature_grid, porosity_grid = np.meshgrid(temperature, porosity)

        num_plots = len(selected_names)
        num_cols = min(3, num_plots)
        num_rows = int(np.ceil(num_plots / num_cols))
        fig, axes = plt.subplots(
            num_rows,
            num_cols,
            squeeze=False,
            constrained_layout=True,
            figsize=(4.8 * num_cols, 3.8 * num_rows),
        )
        axes = axes.ravel()

        for ax, relation_name in zip(axes, selected_names):
            relation, colorbar_label = relations[relation_name]
            values = self._Evaluate_Plot_Relation(
                relation,
                porosity_grid,
                temperature_grid,
            )
            norm = self._Get_Color_Normalisation(values, LogNorm)

            image = ax.pcolormesh(
                temperature_grid,
                porosity_grid,
                values,
                shading="auto",
                norm=norm,
            )
            ax.set_title(relation_name.replace("_", " "))
            ax.set_xlabel("Temperature [C]")
            ax.set_ylabel("Porosity [-]")
            fig.colorbar(image, ax=ax, label=colorbar_label)

        for ax in axes[num_plots:]:
            ax.remove()

        if show:
            plt.show(block=block)
            if not block and draw_pause > 0.0:
                plt.pause(draw_pause)

        used_axes = axes[:num_plots]
        if num_plots == 1:
            return fig, used_axes[0]
        return fig, used_axes

    def _Uses_UFL(self, *values):
        """PRIVATE: Return True when any input is a UFL symbolic expression."""

        return any(isinstance(value, ufl.core.expr.Expr) for value in values)

    def _Exp(self, value, use_ufl):
        """PRIVATE: Evaluate exponential with the numeric or UFL backend."""

        if use_ufl:
            return ufl.exp(value)
        return np.exp(value)

    def _Where_Less_Than(self, value, threshold, true_value, false_value, use_ufl):
        """PRIVATE: Select values with the numeric or UFL backend."""

        if use_ufl:
            return ufl.conditional(ufl.lt(value, threshold), true_value, false_value)
        return np.where(value < threshold, true_value, false_value)

    def _Get_Plot_Relation_Aliases(self, relations):
        """PRIVATE: Build accepted plot-name aliases for relation names."""

        aliases = {}
        for relation_name in relations:
            aliases[self._Normalise_Plot_Relation_Name(relation_name)] = relation_name
            aliases[
                self._Normalise_Plot_Relation_Name(f"Get_{relation_name}")
            ] = relation_name
        return aliases

    def _Normalise_Plot_Relation_Name(self, relation_name):
        """PRIVATE: Normalise user-facing plot relation names."""

        relation_name = str(relation_name).strip().replace(" ", "_").replace("-", "_")
        if relation_name.lower().startswith("get_"):
            relation_name = relation_name[4:]
        return relation_name.lower()

    def _Evaluate_Plot_Relation(self, relation, porosity_grid, temperature_grid):
        """PRIVATE: Evaluate a relation and broadcast scalar outputs."""

        values = np.asarray(relation(porosity_grid, temperature_grid), dtype=float)
        if values.shape == ():
            values = np.full_like(temperature_grid, values, dtype=float)
        return np.broadcast_to(values, temperature_grid.shape)

    def _Get_Color_Normalisation(self, values, LogNorm):
        """PRIVATE: Use log colour scaling for positive multi-decade fields."""

        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            return None
        positive_values = finite_values[finite_values > 0.0]
        if positive_values.size != finite_values.size:
            return None

        vmin = np.min(positive_values)
        vmax = np.max(positive_values)
        if vmin <= 0.0 or vmax <= vmin:
            return None
        if vmax / vmin < 1.0e3:
            return None
        return LogNorm(vmin=vmin, vmax=vmax)
