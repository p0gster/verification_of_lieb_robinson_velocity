from functools import reduce
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm

identity=np.identity(2)
pauli_x=np.array([[0., 1.],
                  [1., 0.]])

pauli_y=np.array([[0., -1j],
                  [1j, 0.]])

pauli_z=np.array([[1., 0.],
                  [0., -1.]])


def reduce_tensor_product(lst):
    #use numpy.kron(a, b)
    return reduce(lambda x,y:np.kron(x,y),lst)

def integer_interval(a,b):

    loop=True
    
    last_element_in_list=a
    list=[a]
    while loop:
        if last_element_in_list >= b:
            loop=False
        else:
            list+=[last_element_in_list+1]
            last_element_in_list=last_element_in_list+1
    return list

def mapl(function,lst):
    return list(map(function,lst))

def boolean_mask(true_or_not_f,lst):
    return mapl(true_or_not_f,lst)

def if_true_matrix(boolean_mask,matrix):
    return mapl(lambda x: identity if x == False else matrix,boolean_mask)

def spin_indexes_to_tensor_product(spin_indexes,pauli_spin_matrix,spins):
    #test spin_indexes_to_tensor_product([2,2],np.matrix([[1,0],[0,-1]]))
    int_interval=integer_interval(1,spins)
    boolean_mask_list=boolean_mask(lambda integer:True if (integer in spin_indexes) else False ,int_interval)
    matrix_list=if_true_matrix(boolean_mask_list,pauli_spin_matrix)
    final_matrix=reduce_tensor_product(matrix_list)

    return final_matrix
    
def sum_function(function,beg_interval,end_interval):
    #adds_a_function over closed interval [beg_interval,end_interval]
    interval=integer_interval(beg_interval,end_interval)
    return reduce(lambda x,y:x+y,map(function,interval),0)

def spin_hamiltonian(N,J,g):
    return -J*sum_function(lambda j:spin_indexes_to_tensor_product([j,j+1],pauli_x,N),1,N-1) \
        + -g* sum_function(lambda j:spin_indexes_to_tensor_product([j],pauli_z,N),1,N)
        


def hamiltonian_eigenvalues_eigenvectors(N,J,g):
    
    # hamiltonian=-J*sum_function(lambda j:spin_indexes_to_tensor_product([j,j+1],pauli_x,N),1,N-1) \
    #     + -g* sum_function(lambda j:spin_indexes_to_tensor_product([j],pauli_z,N),1,N)
    return np.linalg.eig(spin_hamiltonian(N,J,g))

def question_2_lowest_eigenvalues():
    return sorted(hamiltonian_eigenvalues_eigenvectors(7,1,3)[0])[0:4]


def groundstate(eigenvalue_eigenvector_list):
    index_of_minimum_eigenvalue= np.where(eigenvalue_eigenvector_list[0] == eigenvalue_eigenvector_list[0].min())
    groundstate_vector=eigenvalue_eigenvector_list[1][:][index_of_minimum_eigenvalue]
    return np.transpose(groundstate_vector)

def inner_product(vector_1,vector_2):
    #the vectors must be numpy arrays
    return np.matmul(np.transpose(np.conjugate(vector_1)),vector_2)[0][0]
    
def expectation_value(operator,state):
    return inner_product(state ,np.matmul(operator ,state))

def spin_map(state,operator_function,number_of_spins):
    #operator function: an operator that depends on the value of the interval
    return mapl(lambda i:expectation_value(operator_function(i),state),
                integer_interval(1,number_of_spins))

def question_3():
    N=7
    J=1
    g=3
    return spin_map(groundstate(hamiltonian_eigenvalues_eigenvectors(N,J,g)),
                                          lambda j:spin_indexes_to_tensor_product([j],pauli_z,N),
                                          N)

def question_4():
    N=7
    J=1
    g=3
    return spin_map(np.matmul(spin_indexes_to_tensor_product([(N+1)/2],pauli_x,N)
                                                    ,groundstate(hamiltonian_eigenvalues_eigenvectors(N,J,g))),
                                          lambda j:spin_indexes_to_tensor_product([j],pauli_z,N),
                                          N)


def schrodingher_equation_evolve(initial_state,hamiltonian,t):
    return   np.matmul(expm (-1j*t*hamiltonian),initial_state)
    
def float_interval(a,b,step):
    return np.arange(a, b, step).tolist()


def question_6():
    N=7
    J=1
    g=3

    ground_state_vec=groundstate(hamiltonian_eigenvalues_eigenvectors(N,J,g))

    initial_state=np.matmul(spin_indexes_to_tensor_product([(N+1)/2],pauli_x,N)
                            ,ground_state_vec)
    hamiltonian=spin_hamiltonian(N,J,g)
    
    def time_dependant_spin_map(t):
        evolved_state=schrodingher_equation_evolve(initial_state
                                                   ,hamiltonian,t)
        
        return spin_map(evolved_state,
                                              lambda j:spin_indexes_to_tensor_product([j],pauli_z,N),
                                              N)
    return mapl(time_dependant_spin_map,float_interval(0,4,0.1))


def question_6_plot():
   # my data
   spin_colour_array=np.array(question_6()).astype(float)

   plt.imshow(spin_colour_array, cmap='jet',vmin=-1, vmax=1, aspect='auto', extent=[0.5, 7.5, 0, 4], origin='lower')
   plt.colorbar(label='expectation value')
   plt.xlabel('particle')
   plt.xticks(np.arange(1, 8, 1), np.arange(1, 8))
   plt.ylabel('time')
   plt.title('Expectation Values of $\sigma^z$')

   # plt.savefig('spin.pdf',bbox_inches='tight')
   plt.show()
   
   
   
   # plt.style.use('_mpl-gallery-nogrid')

   # # plot
   # fig, ax = plt.subplots()
   # plt.ylabel("time (0.1s)")   
   # plt.xlabel("spins")
   # plt.xticks(np.arange(1, 8, 1), np.arange(1, 8))
   # ax.imshow(spin_colour_array, aspect=0.15)

   # plt.savefig('spin.pdf',bbox_inches='tight')
   # plt.show() 
    
