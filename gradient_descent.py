from firedrake import *
from pyroteus import recover_hessian
from firedrake.meshadapt import adapt, RiemannianMetric
from firedrake.functionspaceimpl import WithGeometry
from firedrake_adjoint import *
from movement import MongeAmpereMover
import numpy as np
import time


def interpolate_between(source, target):
    if source.function_space() == target.function_space():
        target.assign(source)
        return
    from firedrake.mg.utils import physical_node_locations
    target_fs = target.function_space()
    target_locations = physical_node_locations(target_fs).dat.data[:]
    vom = VertexOnlyMesh(source.function_space().mesh(), target_locations, redundant=False)
    if isinstance(target_fs.ufl_element(), VectorElement):
        vom_space = VectorFunctionSpace(vom, "DG", 0)
    else:
        vom_space = FunctionSpace(vom, "DG", 0)
    target_vom_f = interpolate(source, vom_space)
    target.dat.data[:] = target_vom_f.dat.data


def interpolate_to_new_function_space(source, V):
    u = Function(V, name=source.name())
    interpolate_between(source, u)
    return u


def adapt_mesh(mesh, u, q, target_complexity):
    TV = TensorFunctionSpace(mesh, "CG", 1)
    metric = RiemannianMetric(TV).compute_hessian(u)
    metric.rename("metric")
    import numpy as np
    metric.set_parameters(
            {'dm_plex_metric_target_complexity': target_complexity,
             'dm_plex_metric_p': np.inf,
             #  'dm_plex_metric_verbosity': 10,
             'dm_plex_gradation_factor': 1.5,
             'dm_plex_metric_a_max': 10,
             'dm_plex_metric_h_min': 1e-5,
             'dm_plex_metric_h_max': .1})
    H = recover_hessian(u, method='L2')
    metric2 = RiemannianMetric(hessian_metric(H))
    metric.intersect(metric2)
    metric.normalise()
    mesh = adapt(mesh, metric)
    return mesh


class PersistentMeshMover:
    def __init__(self, mesh0, **kwargs):
        self.origx = mesh0.coordinates

        if kwargs.get('debug_files', False):
            self.debug_file = File('movement.pvd')
            self.index = 0
        else:
            self.debug_file = None

        self.phi = None
        self.sigma = None

    def move(self, mesh, u, q):
        # NOTE: currently only adapting to solution u, not to control
        ele = u.function_space().ufl_element()
        TV = TensorFunctionSpace(mesh, "CG", 1)
        H = RiemannianMetric(TV).compute_hessian(u)
        V = FunctionSpace(mesh, "CG", 1)
        Hnorm = interpolate(sqrt(inner(H, H)), V)

        mesh_new = Mesh(Function(self.origx))
        Vnew = FunctionSpace(mesh_new, ele)
        P0new = FunctionSpace(mesh_new, "DG", 0)
        m = Function(Vnew, name='monitor')

        def monitor(mesh):
            m.project(Hnorm)
            mmax = m.dat.data[:].max()
            m.dat.data[:] /= mmax
            m.dat.data[:] = np.maximum(m.dat.data[:], 0.1)
            if self.debug_file:
                self.debug_file.write(m)
                areas = interpolate(CellVolume(mesh), P0new).dat.data[:]
                print("RATIO = ", self.index, areas.max()/areas.min())
                self.index += 1
            return m

        mover = MongeAmpereMover(mesh_new, monitor, method='quasi_newton')
        if self.phi:
            phinew = Function(WithGeometry.create(f.function_space(), mesh), val=phi.topological)
            sigmanew = Function(WithGeometry.create(f.function_space(), mesh), val=sigma.topological)
            mover.apply_initial_guess(phi_init=phinew, sigma_init=sigmanew)

        mover.move()
        return mover.mesh


class DefaultParams:
    max_adapts = 100
    iterations_per_adapt = 5
    initial_target_complexity = 200
    max_complexity = 20000
    initial_learning_rate = 1e-8
    target_increase_factor = 1.5
    move_mesh = False
    debug_movement_files = False


def minimize_gs(run_model, mesh0, ele, params=None, bounds=None):

    p = DefaultParams()
    if params:
        for key, value in params.items():
            setattr(p, key, value)
    if p.move_mesh:
        mover = PersistentMeshMover(mesh0,
                                    debug_files=p.debug_movement_files)

    target_complexity = p.initial_target_complexity

    mesh = mesh0
    logf = open('log.txt', 'w')
    for i in range(p.max_adapts):
        V = FunctionSpace(mesh, ele)
        if i > 0:
            u = interpolate_to_new_function_space(u, V)
            q = interpolate_to_new_function_space(q, V)
            gold = interpolate_to_new_function_space(gold, V)
            qold = interpolate_to_new_function_space(qold, V)
        else:
            u = Function(V)
            q = Function(V)
            gold = Function(V)
            qold = Function(V)
        g = Function(V)
        for j in range(p.iterations_per_adapt):
            tape = get_working_tape()
            tape.clear_tape()
            mesh.create_block_variable()
            c = Control(q)
            J, u = run_model(q, initial_guess=u)
            rf = ReducedFunctional(J, c)
            print(f"i, J = {i, J}")
            g = rf.derivative(options={"riesz_representation": "L2"})
            if i+j == 0:
                alpha = p.initial_learning_rate
            else:
                dq = q - qold
                dg = g - gold
                alpha = np.abs(assemble(dot(dq, dq)*dx)/assemble(dot(dq, dg)*dx))
            qold.assign(q)
            gold.assign(g)
            gnorm = np.sqrt(assemble(dot(g, g)*dx))
            print(f"|g| = {gnorm}, alpha = {alpha}")
            logf.write(f"{i}, {time.process_time()}, {J}, {gnorm}, {alpha}, {len(g.dat.data)}\n")
            logf.flush()
            q.assign(qold - alpha*g)
            if bounds:
                q.interpolate(max_value(bounds[0], min_value(bounds[1], q)))

        with stop_annotating():
            print("ADAPTING THE MESH:")
            if p.move_mesh:
                mesh = mover.move(mesh, u, q)
            else:
                mesh = adapt_mesh(mesh, u, q, target_complexity)
            target_complexity *= p.target_increase_factor
            if target_complexity > p.max_complexity:
                print("Maximum complexity reached")
                break

    return q
