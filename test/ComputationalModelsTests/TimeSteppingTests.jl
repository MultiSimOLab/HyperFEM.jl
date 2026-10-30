# TimeStepping, AdaptiveStepping and the state TimeStep of the transient driver (the control alone, driven by hand).
using HyperFEM
using HyperFEM.ComputationalModels: TimeStep, as_time_stepping, finished, next_target!, accept!, bisect!, reject!, exhausted,
  next_increment, weighted_error, grown_increment
using Test

@testset "TimeStepping" begin
  s = TimeStepping(Δt=0.1, t_end=1.0)
  @test (s.Δt, s.t_end, s.t0, s.maxbisec, s.grow, s.easy, s.Δtmax) == (0.1, 1.0, 0.0, 15, 1.0, 10, 0.1)
  @test s.tstops == Float64[]
  @test as_time_stepping((Δt=0.1, t_end=1.0)) isa TimeStepping
  @test as_time_stepping(s) === s
  @test_throws ErrorException as_time_stepping(4)
  @test_throws ErrorException TimeStepping(Δt=0.0, t_end=1.0)
  @test_throws ErrorException TimeStepping(Δt=0.1, t_end=0.0)
  @test_throws ErrorException TimeStepping(Δt=0.1, t_end=1.0, grow=0.5)
  @test_throws ErrorException TimeStepping(Δt=0.1, t_end=1.0, Δtmax=0.05)
  # tstops: sorted, unique, strictly inside (t0, t_end)
  s = TimeStepping(Δt=0.1, t_end=1.0, tstops=(0.5, 0.25, 0.5, 0.0, 1.0, 2.0))
  @test s.tstops == [0.25, 0.5]

  # the shared growth helper (the floating-point sequence of LoadStepping.step!)
  @test grown_increment(2.0, 10, 0.05, 0.1, true, 3) == 0.1
  @test grown_increment(2.0, 10, 0.03, 0.1, true, 3) == 0.06
  @test grown_increment(2.0, 10, 0.03, 0.1, true, 11) == 0.03
  @test grown_increment(1.0, 10, 0.03, 0.1, true, 3) == 0.03
  @test grown_increment(2.0, 10, 0.03, 0.1, false, 3) == 0.03

  # a run driven by hand: fixed steps cut at t_end
  s = TimeStepping(Δt=0.3, t_end=1.0)
  st = TimeStep(s)
  @test (st.t, st.Δt, st.Δt0, st.nsteps) == (0.0, 0.3, 0.3, 0)
  ts = Float64[]
  while !finished(st, s)
    t, Δt = next_target!(st, s)
    accept!(st, s, 3)
    push!(ts, st.t)
  end
  @test ts ≈ [0.3, 0.6, 0.9, 1.0]
  @test st.Δt_prev ≈ 0.1 && st.Δt == 0.3 && st.nsteps == 4 && st.cut   # the proposed increment survives the cut

  # tstops are reached exactly and the proposed increment is kept after the cut
  s = TimeStepping(Δt=0.3, t_end=1.0, tstops=[0.5])
  st = TimeStep(s)
  ts = Float64[]
  while !finished(st, s)
    next_target!(st, s)
    accept!(st, s, 3)
    push!(ts, st.t)
  end
  @test ts ≈ [0.3, 0.5, 0.8, 1.0]
  @test any(==(0.5), ts)

  # bisection: half of the attempted increment from the same time; growth back after an easy step
  s = TimeStepping(Δt=0.4, t_end=1.0, grow=2.0, easy=5, maxbisec=2)
  st = TimeStep(s)
  next_target!(st, s)
  bisect!(st, s)
  @test st.t == 0.0 && st.Δt == 0.2 && st.nbisect == 1 && st.nbisect_total == 1 && !exhausted(st, s)
  next_target!(st, s); bisect!(st, s)
  @test st.Δt == 0.1 && st.nbisect == 2 && !exhausted(st, s)
  next_target!(st, s); bisect!(st, s)
  @test st.nbisect == 3 && exhausted(st, s)
  st = TimeStep(s)
  next_target!(st, s); bisect!(st, s)
  next_target!(st, s); accept!(st, s, 3)          # easy: grows to 0.4
  @test st.t ≈ 0.2 && st.Δt == 0.4 && st.nbisect == 0 && st.nbisect_total == 1
  next_target!(st, s); bisect!(st, s)
  next_target!(st, s); accept!(st, s, 6)          # not easy: stays halved
  @test st.t ≈ 0.4 && st.Δt == 0.2
  # Δtmax bounds the growth
  s = TimeStepping(Δt=0.1, t_end=1.0, grow=4.0, easy=5, Δtmax=0.15)
  st = TimeStep(s)
  next_target!(st, s); bisect!(st, s)
  next_target!(st, s); accept!(st, s, 1)
  @test st.Δt == 0.15                              # min(4 · 0.05, Δtmax) = 0.15 (the growth is bounded by Δtmax, not by Δt0)
end

@testset "AdaptiveStepping" begin
  s = AdaptiveStepping(Δt=0.1, t_end=1.0)
  @test (s.rtol, s.atol, s.safety, s.fmin, s.fmax, s.controller, s.maxreject, s.maxbisec) == (1e-3, 1e-6, 0.9, 0.2, 2.0, :PI, 20, 15)
  @test s.Δtmin == 1e-8 && s.Δtmax == 1.0
  @test_throws ErrorException AdaptiveStepping(Δt=0.1, t_end=1.0, controller=:PID)
  @test_throws ErrorException AdaptiveStepping(Δt=0.1, t_end=1.0, fmin=1.5)
  @test_throws ErrorException AdaptiveStepping(Δt=0.1, t_end=1.0, Δtmax=0.05)
  @test_throws ErrorException AdaptiveStepping(Δt=0.1, t_end=1.0, safety=0.0)
  @test as_time_stepping(s) === s

  # the controller: I factor, PI factor, clipping, zero error, after a rejection
  s = AdaptiveStepping(Δt=0.1, t_end=10.0, safety=0.9, fmin=0.2, fmax=2.0, Δtmin=1e-3, Δtmax=1.0)
  p = 2
  @test next_increment(s, 0.1, 1.0, NaN, p, false) ≈ 0.09                      # I: safety
  @test next_increment(s, 0.1, 0.9^3, NaN, p, false) ≈ 0.1                     # I: fixed point err = safety^(p+1)
  @test next_increment(s, 0.1, 1e-6, NaN, p, false) ≈ 0.2                      # clipped to fmax
  @test next_increment(s, 0.1, 1e6, NaN, p, false) ≈ 0.02                      # clipped to fmin
  @test next_increment(s, 0.1, 0.0, NaN, p, false) ≈ 0.2                       # zero error: fmax
  @test next_increment(s, 0.9, 1e-6, NaN, p, false) == 1.0                      # Δtmax
  @test next_increment(s, 0.002, 1e6, NaN, p, false) == 1e-3                    # Δtmin
  @test next_increment(s, 0.1, 1e-6, NaN, p, true) == 0.1                       # never grown after a rejection
  @test next_increment(s, 0.1, 4.0, NaN, p, true) ≈ 0.1 * 0.9 * 4.0^(-1 / 3)    # I after a rejection
  kI, kP = 0.3 / 3, 0.4 / 3
  @test next_increment(s, 0.1, 0.5, 0.25, p, false) ≈ 0.1 * 0.9 * 0.5^(-kI) * (0.25 / 0.5)^kP   # PI
  sI = AdaptiveStepping(Δt=0.1, t_end=10.0, controller=:I)
  @test next_increment(sI, 0.1, 0.5, 0.25, p, false) ≈ 0.1 * 0.9 * 0.5^(-1 / 3)

  # a run driven by hand with a model error err = (Δt/Δtref)^(p+1): the controller settles at a fixed
  # point with err < 1, from above (rejections) and from below (growth)
  function run(s, Δtref, p; nmax=200)
    st = TimeStep(s)
    errs = Float64[]
    while !finished(st, s) && !exhausted(st, s) && st.nsteps + st.nreject_total < nmax
      _, Δt = next_target!(st, s)
      err = (Δt / Δtref)^(p + 1)
      if err <= 1
        accept!(st, s, 3, err, p)
        push!(errs, err)
      else
        reject!(st, s, err, p)
      end
    end
    st, errs
  end
  s = AdaptiveStepping(Δt=0.05, t_end=100.0, Δtmin=1e-6, Δtmax=10.0, rtol=1e-3)
  st, errs = run(s, 0.1, 2)                                   # starts below Δtref: grows
  @test st.nreject_total == 0 && st.nsteps > 10
  @test all(<=(1), errs) && abs(errs[end] - errs[end-1]) < 1e-6 && 0.1 < errs[end] < 1
  st, errs = run(s, 0.02, 2)                                  # starts above Δtref: the first attempts are rejected
  @test st.nreject_total >= 1 && st.nreject == 0 && all(<=(1), errs)
  @test st.Δt_prev < 0.02 && abs(errs[end] - errs[end-1]) < 1e-6
  # the I controller settles at err = safety^(p+1)
  sI = AdaptiveStepping(Δt=0.05, t_end=100.0, Δtmin=1e-6, Δtmax=10.0, controller=:I)
  st, errs = run(sI, 0.1, 2)
  @test errs[end] ≈ 0.9^3 rtol = 1e-6

  # rejection restores nothing by itself (the driver does), but bookkeeps and stops after maxreject
  s = AdaptiveStepping(Δt=0.1, t_end=1.0, maxreject=2, Δtmin=1e-9)
  st = TimeStep(s)
  for k in 1:3
    next_target!(st, s)
    reject!(st, s, 8.0, 2)
    @test st.t == 0.0 && st.nreject == k && st.nreject_total == k && st.rejected
  end
  @test exhausted(st, s)
  # Δtmin stops the run
  s = AdaptiveStepping(Δt=0.1, t_end=1.0, Δtmin=0.05, fmin=0.2)
  st = TimeStep(s)
  next_target!(st, s); reject!(st, s, 1e6, 1)
  @test st.Δt == 0.05 && !exhausted(st, s)
  next_target!(st, s); bisect!(st, s)
  @test st.Δt == 0.025 && exhausted(st, s)
  # bisection does not consult the controller and keeps the rejection flag
  s = AdaptiveStepping(Δt=0.1, t_end=1.0)
  st = TimeStep(s)
  next_target!(st, s); bisect!(st, s)
  @test st.Δt == 0.05 && st.nbisect == 1 && st.nreject == 0 && !st.rejected
  # tstops with the adaptive policy: reached exactly, the increment of the controller is based on the cut attempt
  s = AdaptiveStepping(Δt=0.3, t_end=1.0, tstops=[0.5], Δtmax=0.3)
  st = TimeStep(s)
  ts = Float64[]
  while !finished(st, s)
    next_target!(st, s)
    accept!(st, s, 3, 0.5, 2)
    push!(ts, st.t)
  end
  @test 0.5 in ts && ts[end] == 1.0 && issorted(ts)

  # the weighted error: RMS over the masked entries only
  e = [1e-4, 2e-4, 1.0]
  x = [1.0, 2.0, 100.0]
  xn = [0.5, 3.0, 100.0]
  mask = [true, true, false]
  w1 = 1e-6 + 1e-3 * 1.0
  w2 = 1e-6 + 1e-3 * 3.0
  @test weighted_error(e, x, xn, 1e-3, 1e-6, mask) ≈ sqrt(((1e-4 / w1)^2 + (2e-4 / w2)^2) / 2)
  @test weighted_error(e, x, xn, 1e-3, 1e-6, [false, false, false]) == 0.0
end

@testset "Cut steps keep the proposed increment; Δtmax of TimeStepping" begin
  # AdaptiveStepping: a step cut at a tstop does not restart the controller from the cut
  s = AdaptiveStepping(; Δt=0.1, t_end=1.0, rtol=1e-3, tstops=[0.15])
  st = TimeStep(s)
  next_target!(st, s); accept!(st, s, 3, 0.5, 1)          # t = 0.1
  Δt1 = st.Δt
  t, Δt = next_target!(st, s)                             # cut at 0.15
  @test st.cut && Δt ≈ 0.05
  accept!(st, s, 3, 1e-6, 1)
  @test st.Δt >= Δt1                                       # not 0.05 × fmax
  # TimeStepping: the increment grows up to Δtmax, not only back to Δt0
  s = TimeStepping(; Δt=0.1, t_end=10.0, grow=2.0, easy=10, Δtmax=0.3)
  st = TimeStep(s)
  for _ in 1:4
    next_target!(st, s); accept!(st, s, 3)
  end
  @test st.Δt == 0.3
end
