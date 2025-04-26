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
        f.write(',%s\n' % ',,'.join(values2))

        # Rows
        for v, res in zip(values1, results):

            # Values
            calls = (r['calls'] for r in res)
            devs =  (r['deviation'] for r in res)
            vs = ('%i,%.2E' % pair for pair in zip(calls, devs))
            f.write('%s,%s\n' % (v, ','.join(vs)))