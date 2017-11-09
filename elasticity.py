import numpy as np
import matplotlib.pyplot as plt
import sympy as sym



class StressTensor():
    
    def __init__(self, point, expression=None):
        if expression:
            self.evaluate = expression

        self.tensor, self.isSymbol = self.get_tensor(point)
        
    
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
        return λs, vs 
        
        
        
    def principal_tensor(self, point=None):
        λs, _ = self.principal_stress(point) # get principal stresses
        σP = np.diag(λs) # place them them in a diagonal matrix
        return σP
    

    def strain_tensor(self, E, ν, point=None):
        σ, _ = self.get_tensor(point)
        
        σx = σ[0, 0]; σy = σ[1, 1]; σz = σ[2, 2]
        τxy = σ[0, 1]; τxz = σ[0, 2]; τyz = σ[1, 2]
        
        # Generalized Hooke's law
        εx = (σx - ν*(σy + σz))/E
        εy = (σy - ν*(σx + σz))/E
        εz = (σz - ν*(σx + σy))/E

        εxy = (1 + ν)*τxy/E; εxz = (1 + ν)*τxz/E; εyz = (1 + ν)*τyz/E
        
        ε = np.array([[εx, εxy, εxz],
                      [εxy, εy, εyz],
                      [εxz, εyz, εz]])
        
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


    def body_forces(self, point):
        σ, isSymbol = self.get_tensor(point)
        assert isSymbol, "Tensor must be symbolic"

        x, y, z = point
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


    def plot_superficial_forces(self, pA, pB, η, symbols):
        
        vBA = pB - pA

        for i in range(len(vBA)):
            if vBA[i] != 0:
                m = vBA/vBA[i] # slopes
                b = pB - m*pB[i] # bias
                t = symbols[i] # variable
                T = self.stress_vector(η, m*t + b) # stress vector as a function of t
                break

        i = symbols.index(t)

        t_vals = np.sort(np.linspace(pA[i], pB[i], 100))
        T_vals = np.zeros([3, 100])

        for j in range(len(T)):
            Tlamb = sym.lambdify(t, T[j], modules='numpy')
            T_vals[j] = Tlamb(t_vals)
            if T_vals[j].argmax() != T_vals[j].argmin():
                plt.plot(t_vals, T_vals[j])
                plt.xlabel(str(t))
                plt.ylabel('T'+str(symbols[j]))
                plt.grid()
                plt.show()
                print('{} to {}'.format(T_vals[j, 0], T_vals[j, -1]))

        return T


    # def resultant_superficial_forces(self, pA, pB, η, symbols):
        
    #     vBA = pB - pA

    #     for i in range(len(vBA)):
    #         if vBA[i] != 0:
    #             m = vBA/vBA[i] # slopes
    #             b = pB - m*pB[i] # bias
    #             t = symbols[i] # variable
    #             T = self.stress_vector(η, m*t + b) # stress vector as a function of t
    #             break

    #     i = symbols.index(t) # index of the variable

    #     F = np.zeros([len(T), len(T)])
    #     Ct = np.zeros([len(T), len(T)])

    #     for j in range(len(T)):
    #         F[j, j] = sym.integrate(T[j], (t, pA[i], pB[i])) # Resultant superficial force [N/m²]
    #         if F[j, j] != 0:
    #             ct = sym.integrate(t*T[j], (t, pA[i], pB[i]))/F[j, j] # Centroid in the t component
    #             Ct[j] = m*ct + b # Cetroid in all components

    #     return F, Ct, t


    def length_increment(self, E, ν, pA, pB, symbols):
        '''
        For a rect segment pA-pB.
        '''
        σ, isSymbol = self.get_tensor(symbols)
        assert isSymbol, "Tensor must be symbolic"
        x, y, z = symbols

        # Line integral, as in https://www.youtube.com/watch?v=uXjQ8yc9Pdg
        # ∫f(x(t),y(t),z(t))*(dS/dt)*dt, from 0 to l

        vBA = pB - pA
        l = np.linalg.norm(vBA)
        η = vBA/l

        # εn = f(x, y, z)
        ε = self.strain_tensor(E, ν)
        εn = η.dot(ε.dot(η))    
        
        # Parametrization: describing x, y and z as functions of t
        mx = vBA[0]/l; my = vBA[1]/l; mz = vBA[2]/l
        bx = pA[0]; by = pA[1]; bz=pA[2]

        t = sym.Symbol('t', real=True)

        x_t = mx*t + bx
        y_t = my*t + by
        z_t = mz*t + bz

        # dS
        dxdt = sym.diff(x_t, t)
        dydt = sym.diff(y_t, t)
        dzdt = sym.diff(z_t, t)

        dSdt = sym.sqrt(dxdt**2 + dydt**2 + dzdt**2)

        εn_lambd = sym.lambdify([x, y, z], εn, modules='sympy')
        εn_t = εn_lambd(x_t, y_t, z_t)

        # Line integral
        Δl = sym.integrate(εn_t*dSdt, (t, 0, l))

        return float(Δl)


    # def area_increment(self, E, ν, η, pA, pB, point):

    #     # General equation of a plane:
    #     # a*x + b*y + c*z + d = 0
    #     d = -(η[0]*pA[0] + η[1]*pA[1] + η[2]*pA[2])

    #     # creo que si
    #     if η[0] != 0:
    #         ε = σ.strain_tensor(E, ν, (-y*η[1]/η[0] + -z*η[2]/η[0] - d/η[0], y, z))
    #         η1 = np.array([0, 1, 0])
    #         η2 = np.array([0, 0, 1])
    #         t1 = y; t2 = z          
    #     elif η[1] != 0:
    #         ε = σ.strain_tensor(E, ν, (x, -x*η[0]/η[1] + -z*η[2]/η[1] - d/η[1], z))
    #         η1 = np.array([1, 0, 0])
    #         η2 = np.array([0, 0, 1])
    #         t1 = x; t2 = z
    #     elif η[2] != 0:
    #         ε = σ.strain_tensor(E, ν, (x, y, -x*η[0]/η[2] + -y*η[1]/η[2] - d/η[2]))
    #         η1 = np.array([1, 0, 0])
    #         η2 = np.array([0, 1, 0])
    #         t1 = x; t2 = y

    #     εnt1 = η.dot(ε.dot(η))
    #     εnt2 = η.dot(ε.dot(η))

    #     # ya tenemos el campo vectorial (εnt1 + εnt2)
    #     # me faltan los límites de la integral
    #     # eso me lo dan los puntos...

    #     sym.integrate(sym.integrate((εnt1 + εnt2), (t2, 0, -2*y+2)), (t1, 0, 1))




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
            if len(face) == 3:
                
                A, B, C = face

                vBA = B-A 
                vCA = C-A

                γ = np.arccos(vBA.dot(vCA)/(np.linalg.norm(vBA)*np.linalg.norm(vCA)))

                areas[i] = np.linalg.norm(vBA)*np.linalg.norm(vCA)*np.sin(γ)/2

            elif len(face) == 4:
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