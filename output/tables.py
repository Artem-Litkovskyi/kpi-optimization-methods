import os


TABLES_DIR = 'tables'

if not os.path.exists(TABLES_DIR):
    os.makedirs(TABLES_DIR)


def calls_and_deviation(results, values1, values2, subdir, filename):
    dir_path = os.path.join(TABLES_DIR, subdir)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    with open(os.path.join(dir_path, filename + '.csv'), 'w') as f:
        # Header
        f.write('\t%s\n' % '\t\t'.join(values2))

        # Rows
        for v, res in zip(values1, results):

            # Values
            calls = (r['calls'] for r in res)
            devs =  (r['f_deviation'] for r in res)
            vs = ('%i\t%.2E' % pair for pair in zip(calls, devs))
            f.write('%s\t%s\n' % (v, '\t'.join(vs)))


def penalty_method_iters(results_per_iter, subdir, filename):
    dir_path = os.path.join(TABLES_DIR, subdir)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    with open(os.path.join(dir_path, filename + '.csv'), 'w') as f:
        # Header
        f.write('R\tx0\tx*\tf(x*)\tcalls\n')

        # Rows
        for r in results_per_iter:
            last_iter = r[-1]
            # Values
            f.write('%i\t(%.6f; %.6f)\t(%.6f; %.6f)\t%.6f\t%i\n' % (
                last_iter['constraint_r'],
                r[0]['x'][0], r[0]['x'][1],
                last_iter['x'][0], last_iter['x'][1],
                last_iter['f'],
                last_iter['calls']
            ))
