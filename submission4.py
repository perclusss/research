# import modules here
import numpy as np


def output_dense_matrix(M):
    print('\n'.join([' '.join(['{:.6f}'.format(item) for item in row]) for row in M]))


def lf(R, Q, P, max_iter, _err, _mu_1, _mu_2, _lambda_1, _lambda_2):
    num_fac = P.shape[-1]
    num_user = P.shape[0]
    num_item = Q.shape[0]
    Rd = np.zeros([num_item, num_user])
    for i, j, r in R:
        Rd[i, j] = r

    P = P.T

    previous_objective = None

    # print(Rd.shape, P.shape, Q.shape)
    for step in range(max_iter):
        for i in range(num_item):
            for j in range(num_user):
                # print(i, j, Rd[i, j])
                # print(P[i, :])
                # print(Q[:, j])
                eij = Rd[i, j] - np.dot(Q[i, :], P[:, j])
                for k in range(num_fac):
                    Q[i, k] += _mu_1 * 2 * (eij * P[k, j] - _lambda_2 * Q[i, k])
                    P[k, j] += _mu_2 * 2 * (eij * Q[i, k] - _lambda_1 * P[k, j])

        Rd = np.dot(Q, P)
        for i, j, r in R:
            Rd[i, j] = r

        objective = 0.
        for i, j, r in R:
            diff = r - np.dot(Q[i, :], P[:, j])
            objective += diff ** 2

        for i in range(num_item):
            objective += _lambda_2 * np.dot(Q[i, :], Q[i, :])

        for j in range(num_user):
            objective += _lambda_1 * np.dot(P[:, j], P[:, j])

        if previous_objective is not None:
            e = abs(objective - previous_objective)
            print('e = ', e, 'step', step)
            if e < _err:
                print('break')
                break
        previous_objective = objective
    return Q, P.T


def run():
    num_item = 4
    num_user = 4
    matrix_R = np.loadtxt('asset/data.txt', dtype=int)
    ran_generator = np.random.RandomState(24)
    num_fac = 9

    matrix_Q = ran_generator.random_sample((num_item, num_fac))
    matrix_P = ran_generator.random_sample((num_user, num_fac))

    Q, P = lf(matrix_R, matrix_Q, matrix_P, 2000, 0.0001, 0.01, 0.01, 0.1, 0.1)
    print('========== Q ==========')
    output_dense_matrix(Q)
    print('========== P ==========')
    output_dense_matrix(P)


if __name__ == '__main__':
    run()
