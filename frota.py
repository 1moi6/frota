import numpy as np
import random
from itertools import product
from joblib import Parallel, delayed, effective_n_jobs


class Frota:
    def __init__(self, Periodo, TamanhoFrota, Demanda):
        while len(Demanda) < Periodo+1:
            Demanda.append(random.randint(int(0.45*TamanhoFrota), TamanhoFrota))

        self.ProbQuebra_0 = 0.01
        self.Depreciacao = 0.4
        self.ProbPar = 0.04
        self.Periodo = Periodo
        self.TamanhoFrota = TamanhoFrota
        self.Demanda = Demanda
        self.ProbQuebraFcn = lambda x: 0 if x == 0 else 1 - \
            (1 - self.ProbQuebra_0) * np.exp(-self.ProbPar * x)
        self.Estados = {}
        self.Acoes = {}
        self.Politica = []
        self.__num_estados__ = (Periodo+1)**TamanhoFrota
        self.__num_acoes__ = 2**TamanhoFrota
        self.ConstroeEstadosP()
        self.ConstroeAcoes()
        
    def num2coefs(self, numero, base):
        coeficientes = [0]*self.TamanhoFrota
        if numero == 0:
            return coeficientes

        cnt = 0
        while numero > 0:
            coeficiente = numero % base
            # Insere o coeficiente à esquerda da lista
            coeficientes[cnt] = coeficiente
            numero //= base  # Divide o número pelo valor da base
            cnt = cnt+1

        return coeficientes

    def ConstroeAcoes(self):
        acoes = {}
        acoes[-1] = [-1]
        for i in range(self.Periodo+1):
            acoes[i] = []
            for j in range(self.__num_acoes__):
                coefs = self.num2coefs(j, 2)
                aux = self.TamanhoFrota-sum(coefs)
                if aux >= self.Demanda[i]:
                    acoes[i].append(j)

        self.Acoes = acoes

    def ConstroeEstadosP(self):
        tam_partes= min(int(self.__num_estados__/effective_n_jobs()), 10000)
        
        estados = {}
        estados[(0, 0)] = 0

        def worker(start, end):
            estados_locais = {}
            for idx in range(start, end):
                coefs = self.num2coefs(idx, self.Periodo+1)
                midx = max([*coefs, 1])
                for j in range(midx, self.Periodo+1):
                    estados_locais[(idx, j)] = 0

            return estados_locais

        intervals = [(i, min(i + tam_partes, self.__num_estados__))
                     for i in range(0, self.__num_estados__, tam_partes)]
        results = Parallel(n_jobs=-1)(delayed(worker)(start, end)
                                      for start, end in intervals)

        for result in results:
            estados.update(result)

        estados[(-1, -1)] = 0
        self.Estados = estados

    def ConstroeEstados3(self):
        estados = {}
        estados[(0, 0)] = -1
        estados[(-1, -1)] = 0

        for idx in range(self.__num_estados__):
            coefs = self.num2coefs(idx, self.Periodo+1)
            midx = max([*coefs, 1])
            for j in range(midx, self.Periodo+1):
                estados[(idx, j)] = -1
        self.Estados = estados

    def Recompensa(self, v, p):
        tvida = 0

        # define o valor do equipamento como função da idade do equipamento
        valor_equipamento = np.exp(-0.005*tvida)

        # define o valor da manutenção
        valor_manutencao = 0.02/valor_equipamento

        # define o valor da manutenção se der defeito
        valor_defeito = 2*valor_manutencao

        # Cálculo das recompensas SEM defeito e COM defeito
        res_a0_op, res_a0_de = 0, -valor_defeito

        # Cálculo das recompensas na manutenção
        res_a1 = -valor_manutencao

        res = [res_a0_op, res_a0_de, res_a1]
        if p == -1:
            res = [0, 0, 0]
        return res

    def coefs2num(self, coefs, base):
        num = 0
        for i, j in enumerate(coefs):
            num = num+j*base**i

        return num

    def ProximoEstado(self, estado):
        p = estado[-1]
        est = estado[0]
        acoes = self.Acoes[p]
        NovosEstados = {"estados": [], "recompensas": [], "probabilidades": []}
        coefs = self.num2coefs(est, self.Periodo+1)
        if p in [self.Periodo, -1]:

            return {"estados": [[(-1, -1)]]*len(acoes), "recompensas": [[0]]*len(acoes), "probabilidades": [[1]]*len(acoes)}
        for ac in acoes:
            ac_cf = self.num2coefs(ac, 2)
            NovoEstado = [[]]*self.TamanhoFrota
            Recompensa = [[]]*self.TamanhoFrota
            Probabilidades = [[]]*self.TamanhoFrota
            for i, a in enumerate(ac_cf):
                rcp_aux = self.Recompensa(coefs[i], p)
                if a == 0:
                    NovoEstado[i] = [0, coefs[i]+1]
                    prob_quebrar = self.ProbQuebraFcn(coefs[i])
                    Probabilidades[i] = [prob_quebrar, 1-prob_quebrar]
                    Recompensa[i] = [rcp_aux[1], rcp_aux[0]]
                if a == 1:
                    NovoEstado[i] = [0]
                    Probabilidades[i] = [1]
                    Recompensa[i] = [rcp_aux[2]]
                if a == -1:
                    NovoEstado[i] = [-1]
                    Probabilidades[i] = [1]
                    Recompensa[i] = [0]

            probs = list(product(*Probabilidades))
            recs = list(product(*Recompensa))
            probs = [np.prod(pr) for pr in probs]
            recs = [np.sum(rc) for rc in recs]
            NovosEstados["estados"].append([(self.coefs2num(
                x, self.Periodo+1), estado[-1]+1) for x in list(product(*NovoEstado))])
            NovosEstados["recompensas"].append(recs)
            NovosEstados["probabilidades"].append(probs)

        return NovosEstados

    def Proximos(self, estado, acao):
        p = estado[-1]
        est = estado[0]
        NovosEstados = {"estados": [], "recompensas": [], "probabilidades": []}
        coefs = self.num2coefs(est, self.Periodo+1)
        if p in [self.Periodo, -1]:

            return {"estados": [(-1, -1)], "recompensas": [0], "probabilidades": [1]}
        ac_cf = self.num2coefs(acao, 2)
        NovoEstado = [[]]*self.TamanhoFrota
        Recompensa = [[]]*self.TamanhoFrota
        Probabilidades = [[]]*self.TamanhoFrota
        for i, a in enumerate(ac_cf):
            rcp_aux = self.Recompensa(coefs[i], p)
            if a == 0:
                NovoEstado[i] = [0, coefs[i]+1]
                prob_quebrar = self.ProbQuebraFcn(coefs[i])
                Probabilidades[i] = [prob_quebrar, 1-prob_quebrar]
                Recompensa[i] = [rcp_aux[1], rcp_aux[0]]
            if a == 1:
                NovoEstado[i] = [0]
                Probabilidades[i] = [1]
                Recompensa[i] = [rcp_aux[2]]
            if a == -1:
                NovoEstado[i] = [-1]
                Probabilidades[i] = [1]
                Recompensa[i] = [0]

        probs = list(product(*Probabilidades))
        recs = list(product(*Recompensa))
        probs = [np.prod(pr) for pr in probs]
        recs = [np.sum(rc) for rc in recs]
        NovosEstados["estados"] = [(self.coefs2num(
            x, self.Periodo+1), estado[-1]+1) for x in list(product(*NovoEstado))]
        NovosEstados["recompensas"] = recs
        NovosEstados["probabilidades"] = probs

        return NovosEstados

    def ValueIteration(self, tol=1.0e-10, nsim=500):
        cnt, D, norms, sim = 0, 10, [], 0

        self.Politica = {est: 0 for est in list(self.Estados)}
        while (D > tol and sim < nsim):
            D = 0
            V = {e: v for e, v in self.Estados.items()}
            for est, val in self.Estados.items():
                ac = self.Acoes[est[-1]]
                next = self.ProximoEstado(est)
                nst = next["estados"]
                prs = next["probabilidades"]
                rcs = next["recompensas"]
                v_aux = []

                for ja, a in enumerate(ac):
                    v_aux2 = 0
                    for je, e in enumerate(nst[ja]):
                        v_aux2 = v_aux2 + prs[ja][je]*(rcs[ja][je]+V[e])
                    v_aux.append(v_aux2)

                self.Politica[est] = ac[np.argmax(v_aux)]
                v_aux = np.max(v_aux)
                D = max([D, np.abs(v_aux-val)])
                self.Estados[est] = v_aux

            cnt = cnt+1
            norms.append(D)

        return norms

    def ValueIterationP(self, tol=1.0e-10, nsim=500):
        sim, D, norms, sim = 0, 10, [], 0

        tam_partes= min(int(len(self.Estados)/effective_n_jobs()), 10000)

        self.Politica = {est: 0 for est in list(self.Estados)}

        def particionar_dicionario(dicionario, tamanho_parte):
            chaves = list(dicionario.keys())
            valores = list(dicionario.values())

            particionado = []
            for i in range(0, len(chaves), tamanho_parte):
                particionado.append({chaves[j]: valores[j] for j in range(
                    i, min(i + tamanho_parte, len(chaves)))})

            return particionado

        def loop(V0):
            V1 = {}
            P1 = {}
            D1 = 0
            for est, val in V0.items():
                ac = self.Acoes[est[-1]]
                next = self.ProximoEstado(est)
                nst = next["estados"]
                prs = next["probabilidades"]
                rcs = next["recompensas"]
                v_aux = []

                for ja, a in enumerate(ac):
                    v_aux2 = 0
                    for je, e in enumerate(nst[ja]):
                        v_aux2 = v_aux2 + prs[ja][je]*(rcs[ja][je]+self.Estados[e])
                    v_aux.append(v_aux2)

                P1[est] = ac[np.argmax(v_aux)]
                v_aux = np.max(v_aux)
                D1 = max([D1, np.abs(v_aux-val)])
                V1[est] = v_aux
            return V1, P1, D1

        while (D > tol and sim < nsim):
            # V = {e: v for e, v in self.Estados.items()}
            partes = particionar_dicionario({e: v for e, v in self.Estados.items()}, tam_partes)
            results = Parallel(n_jobs=-1)(delayed(loop)(prt) for prt in partes)

            D = 0
            for result in results:
                V1, P1, D1 = result
                self.Estados.update(V1)
                self.Politica.update(P1)
                D = max(D, D1)

            sim = sim+1
            norms.append(D)

            print(f"sim: {sim:3d}, D: {D:6.4f}")

        return norms

    def CaminhoProvavel(self):
        estado = (0, 0)
        coefs = self.num2coefs(estado[0], self.Periodo+1)
        veiculos = [f"v_{i+1}" for i in range(self.TamanhoFrota)]
        caminho = {veiculos[i]: [t] for i, t in enumerate(coefs)}
        while estado[-1] < self.Periodo:
            aux = self.Proximos(estado, self.Politica[estado])
            ac_id = np.argmax(aux["probabilidades"])
            estado = aux["estados"][ac_id]
            coefs = self.num2coefs(estado[0], self.Periodo+1)
            for i, v in enumerate(veiculos):
                caminho[v].append(coefs[i])

        return caminho
