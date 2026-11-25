# Staggered (partitioned) models

A coupled problem can also be solved field by field: `StaggeredModel` takes one computational model per
field (`StaticNonlinearModel`s), the states the models solve for and their lagged copies, through which the
forms and the `Coupling` conditions of a field read the other fields. The loop advances a stage τ from 0
to 1 by steps; in every step the models are solved one after another (a *pass*), each over its own inner
loop with the load of the step, and the lagged states are set to the new ones after every pass:

```julia
# electrics with the lagged deformation, mechanics with the new potential (uh⁺, φh⁺: states; uh⁻, φh⁻: lagged)
mech(Λ) = uh⁻ + (uh⁺ - uh⁻) * Λ                      # Λ: stage of the inner solve, 0 → 1 within a step
res_elec(Λ) = (φ, vφ) -> residual(physmodel, Electro, k, (mech(Λ), φ), vφ, dΩ)
res_mec(Λ)  = (u, v)  -> residual(physmodel, Mechano, k, (u, φh⁺), v, dΩ)
m_elec = StaticNonlinearModel(res_elec, jac_elec, Uφ, Vφ, Dφ; nls, xh=φh⁺)
m_mec  = StaticNonlinearModel(res_mec,  jac_mec,  Uu, Vu, Du; nls, xh=uh⁺)
sm = StaggeredModel((m_elec, m_mec), (φh⁺, uh⁺), (φh⁻, uh⁻))
args = ((stepping=(nsteps=1, maxbisec=2),), (stepping=(nsteps=1, maxbisec=2),))   # keywords of each inner solve!

# fixed loop: 100 steps of one pass each (the result depends on the number of steps)
x, flag = solve!(sm; stepping=StaggeredStepping(nsteps=100), kargsolve=args)

# coupling converged in every step: few steps, passes until the relative increment of every field is ≤ 1e-6
# (quasi-Newton update of the displacement; a failed step is bisected, an easy one lets the increment grow)
x, flag = solve!(sm; stepping=StaggeredStepping(nsteps=10, max_substeps=30, coupling_tol=1e-6,
                                                relaxation=(:none, IQNILS()), grow=2.0, easy=4), kargsolve=args,
                 presolver=(τ⁻, Δτ) -> ramp!(τ⁻ + Δτ),   # a load ramped by the loop reads the stage of the step
                 on_step=(τ, Δτ, nsub) -> @info "τ = $τ in $nsub passes")
log = get_log(sm)          # τ, Δτ, passes, coupling errors, relaxation factors and times per step
```

With `coupling_tol = nothing` (the default) the loop is the fixed one of earlier versions: `nsteps` equal steps
of `max_substeps` passes, whatever the convergence (the `NamedTuple` `(nsteps=…, nsubsteps=…, maxbisec=…)`
is still accepted and selects it). With a tolerance, the passes of a step stop when every field is converged
(`norm = :increment`, the relative increment of the free values of the field in the pass; `norm = :residual`,
the residual of the field at the new state of the others relative to the one of the first pass); a step whose
passes reach `max_substeps` without convergence, or in which an inner solve fails, is rejected: the states, the
lagged states and the Dirichlet values return to the last accepted step and the increment is bisected
(`maxbisec` consecutive bisections at most); after `neasy` consecutive steps accepted in at most `easy` passes
each (one, by default) the increment grows by `grow`, up to `Δτmax` (the initial increment by default).
`relaxation` is `:none`, a factor, `:aitken`/`Aitken(; ω0, ωmin, ωmax)`
(`ωmax > 1` allows over-relaxation, which a monotonically converging iteration needs to gain anything),
`:iqnils`/`IQNILS()` (a quasi-Newton update of the coupling built from the passes of the step: the choice when
the coupling is stiffer than a field's own stiffness, where the passes oscillate and no scalar factor converges),
or a tuple with one of them per field; the factor of a field whose model has an `injectivity` measure and an
`InjectivityLS` is cut so that the relaxed state does not invert elements. The post-processors of the inner
models run inside every pass; output per accepted step belongs in `on_step`.

## Fields of other problems: `Coupling`, `StageBlend` and `Transfer`

A `Coupling(g)` prescribes on a boundary the state of another problem, `g(τ)` at the stage τ of the solver. The
field may live on another mesh, on the volume next to the boundary or only on the faces of the tag: the nodes of
the tag are located in the mesh of the field once, when the model builds its Dirichlet applicator, and no point
search is done afterwards. A finite-element function (or a `StageBlend` of two) is then transferred through its
dof values — a copy when the nodes coincide, a sparse product otherwise — and any other field is evaluated in the
located cells. Fields that are not Dirichlet data (loads, tractions) go between two spaces with a `Transfer`:

```julia
# the vacuum follows the solid on their interface (staggered: the solid between its two stages)
uhsolid = StageBlend(uh_solid⁻, uh_solid⁺)       # uhsolid(τ) = uh_solid⁻ + (uh_solid⁺ - uh_solid⁻) * τ, also in the weak forms
Du_vacuum = BoundaryConditions(BoundaryCondition("Fixed_Air", [0.0, 0.0]),
                               BoundaryCondition("Interface", Coupling(uhsolid)))    # match = :auto, tol = 1e-10

# nodal forces of the vacuum on the interface of the solid
T = Transfer(V_maxwell, Usolid_int)              # built once
transfer!(tmaxw_, T, Pmaxw)                      # values at the dofs of Usolid_int (a copy here: the nodes coincide)
transfer_transpose!(f, T, g)                     # conservative transfer of loads (the resultant is kept)
```

`match = :auto` matches coincident nodes exactly and searches the others in the cells of the source mesh;
`match = :search` always searches. A node that is not on the source mesh is an error, never an extrapolation.
The application is linear algebra on flat arrays (no allocations, generic in the array type), the part that
carries over to distributed and GPU arrays; a `Coupling` whose `g(τ)` is not a finite-element function or a
`StageBlend` is evaluated by Gridap in the located cells. Gridap's `Interpolable` is still accepted.
