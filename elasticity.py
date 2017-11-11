import numpy as np
import matplotlib.pyplot as plt
import sympy as sym



class StressTensor():
    
    def __init__(self, symbols=None, expression=None):
        if expression:
            self.evaluate = expression

        if symbols:
            self.symbols = symbols # x, y, z
            self.tensor, self.isSymbol = self.get_tensor(symbols)
        
    
    @staticmethod   
    def evaluate(point):
        σ = np.array([[1, 0, 0], 
                  [0, 1, 0], 
                  [0, 0, 1]])
        return σ
        
    def get_tensor(self, point=None):
        if point is None:
            σ = self.tensor
            isSymbol = self.isSymbol
        else:
            σ = self.evaluate(point)
            isSymbol = False
            for c in point:
                if hasattr(c, 'free_symbols'):
                    isSymbol = True
            if not isSymbol: σ = σ.astype(np.float32)
                    
        return σ, isSymbol
            
        
    def principal_stress(self, point=None):
        σ, isSymbol = self.get_tensor(point)
        assert not isSymbol, "Unable to find eigenvalues to symbolic tensor. Please, input a real point"
        λs, vs = np.linalg.eig(σ) # get eigenvalues and eigenvectors
        vs = vs[np.argsort(λs)[::-1]] # sort eigenvectors
        λs = np.sort(λs)[::-1] # sort eigenvalues
        return λs, vs.T
        
        
        
    def principal_tensor(self, point=None):
        λs, _ = self.principal_stress(point) # get principal stresses
        σP = np.diag(λs) # place them them in a diagonal matrix
        return σP
    

    def strain_tensor(self, E, ν, point=None):
        '''
        Returns a StrainTensor object
        '''
        σ, _ = self.get_tensor(point)
        
        σx = σ[0, 0]; σy = σ[1, 1]; σz = σ[2, 2]
        τxy = σ[0, 1]; τxz = σ[0, 2]; τyz = σ[1, 2]
        
        # Generalized Hooke's law
        εx = (σx - ν*(σy + σz))/E
        εy = (σy - ν*(σx + σz))/E
        εz = (σz - ν*(σx + σy))/E

        εxy = (1 + ν)*τxy/E; εxz = (1 + ν)*τxz/E; εyz = (1 + ν)*τyz/E
        
        ε = StrainTensor(self.symbols)
        ε.tensor = np.array([[εx, εxy, εxz],
                  [εxy, εy, εyz],
                  [εxz, εyz, εz]])

        def gen_ε(self, point):
            ε_sym = self.tensor
            x, y, z = self.symbols

            ε = np.zeros([9]).astype('object')
            for i, εi in enumerate(ε_sym.reshape([-1])):
                εi_lamb = sym.lambdify([x, y, z], εi)
                ε[i] = σi_lamb(*point)

            ε = ε.reshape([3, 3])
            return ε

        from functools import partial
        ε.evaluate = partial(gen_ε, ε)

        return ε
    
    
    def unit_vol_increment(self, E, ν, point=None):
        σ, _ = self.get_tensor(point)
        e = np.trace(σ)*(1 - 2*ν)/E
        return e
    
    
    def stress_vector(self, η, point=None):
        σ, _ = self.get_tensor(point)
        T = σ.dot(η) # Cauchy's law
        return T
    
    
    def stress_vector_components(self, η, point=None):
        _, isSymbol = self.get_tensor(point)
        T = self.stress_vector(η, point)
        σn = η.T.dot(T)
        if isSymbol:
            τ = sym.sqrt((T**2).sum() - σn**2)
        else:
            τ = np.sqrt(np.array([(T**2).sum() - σn**2]))
        return σn, τ

    
    def invariants(self, point=None):
        σ, isSymbol = self.get_tensor(point)
        I1 = np.trace(σ)
        I2 = ((np.trace(σ))**2 - np.trace(σ.dot(σ)))/2
        
        if isSymbol:
            print("Unable to compute third invariant for symbolic tensor")
            return I1, I2, False
        else:
            I3 = np.linalg.det(σ)
            return I1, I2, I3
        

    def von_mises_stress(self, point=None):
        I1, I2, isNotSymbol = self.invariants(point)
        if isNotSymbol:
            σeq = np.sqrt(I1**2 - 3*I2)
        else:
            σeq = sym.sqrt(I1**2 - 3*I2)
        return σeq

    
    def draw_mohrs_circles(self, point):
        fig = plt.figure()
        
        σP, _ = self.principal_stress(point)
        
        θ = np.linspace(0, 2*np.pi, 100)
        
        # Arrays storing the radii and centers of the 3 circles
        r = np.zeros([3, 1])
        c = np.zeros([3, 1])
        
        σs = np.zeros([3, 100])
        τs = np.zeros([3, 100])
        
        for i in range(3):
            r[i] = np.abs(σP[-i] - σP[1-i])/2
            c[i] = (σP[-i] + σP[1-i])/2
            
            σs[i] = r[i]*np.cos(θ) + c[i]
            τs[i] = r[i]*np.sin(θ)
            
            plt.plot(σs[i], τs[i])
            
        plt.axis('equal')
        plt.grid(True)
        
        return fig


    def body_forces(self):
        σ, isSymbol = self.get_tensor()
        assert isSymbol, "Tensor must be symbolic"

        x, y, z = self.symbols
        σx = σ[0, 0]; σy = σ[1, 1]; σz = σ[2, 2]
        τxy = σ[0, 1]; τxz = σ[0, 2]; τyz = σ[1, 2]

        # Equilibrium of internal body forces
        X = -(sym.diff(σx, x) + sym.diff(τxy, y) + sym.diff(τxz, z))
        Y = -(sym.diff(τxy, x) + sym.diff(σy, y) + sym.diff(τyz, z))
        Z = -(sym.diff(τxz, x) + sym.diff(τxz, y) + sym.diff(σz, z))

        X = np.array([float(X), 0, 0])
        Y = np.array([0, float(Y), 0])
        Z = np.array([0, 0, float(Z)])

        return X, Y, Z


    def plot_superficial_forces_2D(self, pA, pB, η):
        
        vBA = pB - pA

        for i in range(len(vBA)):
            if vBA[i] != 0:
                m = vBA/vBA[i] # slopes
                b = pB - m*pB[i] # bias
                t = self.symbols[i] # variable
                T = self.stress_vector(η, m*t + b) # stress vector as a function of t
                break

        i = self.symbols.index(t)

        t_vals = np.sort(np.linspace(pA[i], pB[i], 100))
        T_vals = np.zeros([3, 100])

        for j in range(len(T)):
            Tlamb = sym.lambdify(t, T[j], modules='numpy')
            T_vals[j] = Tlamb(t_vals)
            if T_vals[j].argmax() != T_vals[j].argmin():
                plt.plot(t_vals, T_vals[j])
                plt.xlabel(str(t))
                plt.ylabel('T'+str(self.symbols[j]))
                plt.grid()
                plt.show()
                print('{} to {}'.format(T_vals[j, 0], T_vals[j, -1]))

        return T



class Polygon():
    
    def __init__(self, faces):
        self.faces = faces
        
        
    def normal_vectors(self):
        ηs = np.zeros([len(self.faces), 3])
        for i, face in enumerate(self.faces):
            vs = face[0] - face[1:] # vectors from first point to any of the others
            vs = vs[:2] # only take the first two vectors
            n = np.cross(*vs) # cross product of these two vectors to get the normal vector of the face
            ηs[i] = n/np.linalg.norm(n) # make the normal vector a unit vector
        return ηs
        
        
    def draw(self, in_window=False):
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        fig = plt.figure()
        ax = Axes3D(fig)
        
        lmax = 0; lmin = 0
        for face in self.faces:
            farr = np.array(face)
            if lmax < farr.max(): lmax = farr.max()
            if lmin > farr.min(): lmin = farr.min()

        ax.set_xlim3d(lmin, lmax)
        ax.set_ylim3d(lmin, lmax)
        ax.set_zlim3d(lmin, lmax)
        
        pol = Poly3DCollection(self.faces)
        pol.set_color('c')
        pol.set_edgecolor('k')
        ax.add_collection3d(pol)
        
        if not in_window:
            plt.show()


    def areas(self):
        areas = np.zeros(len(self.faces))

        for i, face in enumerate(self.faces):
            if len(face) == 3: # triangle
                
                A, B, C = face

                vBA = B-A 
                vCA = C-A

                γ = np.arccos(vBA.dot(vCA)/(np.linalg.norm(vBA)*np.linalg.norm(vCA)))

                areas[i] = np.linalg.norm(vBA)*np.linalg.norm(vCA)*np.sin(γ)/2

            elif len(face) == 4: # quadrilateral
                # Area of quadrilateral using Bretschneider's formula

                A, B, C, D = face

                vBA = B-A   
                vDA = D-A
                vDC = D-C
                vBC = B-C
                
                a = np.linalg.norm(vBA)
                b = np.linalg.norm(vDA)
                c = np.linalg.norm(vDC)
                d = np.linalg.norm(vBC)
                s = (a+b+c+d)/2

                α = np.arccos(vBA.dot(vDA)/(np.linalg.norm(vBA)*np.linalg.norm(vDA)))
                γ = np.arccos(vDC.dot(vBC)/(np.linalg.norm(vDC)*np.linalg.norm(vBC)))

                areas[i] = np.sqrt((s-a)*(s-b)*(s-c)*(s-d) - a*b*c*d*(1 + np.cos(α+γ))/2)
            else:
                print("Unable to calculate area for faces with more than 4 points.")

        return areas




class DisplacementField():

    def __init__(self, symbols, expression):
        self.symbols = symbols
        self.evaluate = expression
        self.field = self.evaluate(symbols)


    @staticmethod
    def evaluate(point):
        u = np.array([1,
            1,
            1])
        return u


    def strain_tensor(self):
        '''
        Returns a StrainTensor object
        '''
        u = self.field
        x, y, z = self.symbols

        εxx = sym.diff(u[0], x)
        εyy = sym.diff(u[1], y)
        εzz = sym.diff(u[2], z)

        εxy = (sym.diff(u[1], x) + sym.diff(u[0], y))/2
        εxz = (sym.diff(u[2], x) + sym.diff(u[0], z))/2
        εyz = (sym.diff(u[2], y) + sym.diff(u[1], z))/2

        ε = StrainTensor(self.symbols)
        ε.tensor = np.array([[εxx, εxy, εxz],
                  [εxy, εyy, εyz],
                  [εxz, εyz, εzz]])

        def gen_ε(self, point):
            ε_sym = self.tensor
            x, y, z = self.symbols

            ε = np.zeros([9]).astype('object')
            for i, εi in enumerate(ε_sym.reshape([-1])):
                εi_lamb = sym.lambdify([x, y, z], εi)
                ε[i] = σi_lamb(*point)

            ε = ε.reshape([3, 3])
            return ε

        from functools import partial
        ε.evaluate = partial(gen_ε, ε)

        return ε




class StrainTensor():

    def __init__(self, symbols=None, expression=None):
        if expression:
            self.evaluate = expression

        if symbols:
            self.symbols = symbols # x, y, z
            self.tensor, self.isSymbol = self.get_tensor(symbols)


    @staticmethod
    def evaluate(point):
        ε = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])
        return ε


    def get_tensor(self, point=None):
        if point is None:
            ε = self.tensor
            isSymbol = self.isSymbol
        else:
            ε = self.evaluate(point)
            isSymbol = False
            for c in point:
                if hasattr(c, 'free_symbols'):
                    isSymbol = True
            if not isSymbol: ε = ε.astype(np.float32)

        return ε, isSymbol


    def stress_tensor(self, E, ν):
        '''
        Returns a StressTensor object
        '''
        ε, isSymbol = self.get_tensor()
        assert isSymbol, "Tensor must be symbolic"

        σ = StressTensor(self.symbols)

        # Lamé equations
        G = E/(2*(1 + ν))
        λ = 2*G*ν/(1 - 2*ν)
        σ.tensor = 2*G*ε + λ*np.trace(ε)*np.eye(3)

        def gen_σ(self, point):
            σ_sym = self.tensor
            x, y, z = self.symbols

            σ = np.zeros([9]).astype('object')
            for i, σi in enumerate(σ_sym.reshape([-1])):
                σi_lamb = sym.lambdify([x, y, z], σi)
                σ[i] = σi_lamb(*point)

            σ = σ.reshape([3, 3])
            return σ

        from functools import partial
        σ.evaluate = partial(gen_σ, σ)

        return σ


    def length_increment(self, pA, pB):
        '''
        For a rect segment pA-pB.
        '''
        ε, isSymbol = self.get_tensor()
        assert isSymbol, "Tensor must be symbolic"
        x, y, z = self.symbols

        # Line integral, as in https://www.youtube.com/watch?v=uXjQ8yc9Pdg
        # ∫f(x(t),y(t),z(t))*|rt|*dt, from 0 to l

        vBA = pB - pA
        l = np.linalg.norm(vBA)
        η = vBA/l

        # εn = εn(x, y, z)
        εn = η.dot(ε.dot(η))

        # Parametrization: describing x, y and z as functions of t
        t = sym.Symbol('t', real=True)

        x_t = η[0]*t + pA[0]
        y_t = η[1]*t + pA[1]
        z_t = η[2]*t + pB[2]

        # dS = |rt|*dt
        dSdt = np.linalg.norm(η)

        # Describing εn as a function of t
        εn_lambda = sym.lambdify([x, y, z], εn, modules='sympy')
        εn_t = εn_lambda(x_t, y_t, z_t)

        # Line integral
        Δl = sym.integrate(εn_t*dSdt, (t, 0, l))

        return float(Δl)


    def rect_area_increment(self, pA, pB, pC):
        '''
        For a rectangular area defined by sides AB and AC.
        '''
        ε, isSymbol = self.get_tensor()
        assert isSymbol, "Tensor must be symbolic"
        x, y, z = self.symbols

        # Surface integral, as in https://www.youtube.com/watch?v=tyVCA_8MUV4
        # ∫f(x(u,v),y(u,v),z(u,v))*|ru x rv|*dudv, 0<=u<=lu, 0<=v<=lv

        vAB = pB - pA; vAC = pC - pA
        lAB = np.linalg.norm(vAB); lAC = np.linalg.norm(vAC)
        ηu = vAB/lAB; ηv = vAC/lAC

        # εu = εu(x, y, z); εv = εv(x, y, z)
        εu = ηu.dot(ε.dot(ηu))
        εv = ηv.dot(ε.dot(ηv))

        # Parametrization: describing x, y and z as functions of u and v
        u, v = sym.symbols('u, v', real=True)

        x_uv = ηu[0]*u + ηv[0]*v + pA[0]
        y_uv = ηu[1]*u + ηv[1]*v + pA[1]
        z_uv = ηu[2]*u + ηv[2]*v + pA[2]

        # dS = |ru x rv|*dt
        dSdt = np.linalg.norm(np.cross(ηu, ηv))

        # Describing εu and εv as functions of u and v
        εu_lambda = sym.lambdify([x, y, z], εu, modules='sympy')
        εu_uv = εu_lambda(x_uv, y_uv, z_uv)
        εv_lambda = sym.lambdify([x, y, z], εv, modules='sympy')
        εv_uv = εv_lambda(x_uv, y_uv, z_uv)

        # Surface integral
        ΔA = sym.integrate((εu_uv + εv_uv)*dSdt, (u, 0, lAB), (v, 0, lAC))

        return float(ΔA)
