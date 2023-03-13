import matplotlib.pyplot as plt

def preprocess_data(df):
    df['Q'] = df.Q.str.replace(',', '.').astype(float)
    df['Tienda'] = df['Tienda'].astype(str)
    df = df.sort_values('Ruta')
    return df

def export_result(df, best_solution):
    df['station'] = 0
    for i, station in enumerate(best_solution):
        for shop in station:
            df.loc[df['Tienda'] == shop, 'station'] = i+1
            # print(i, shop, costs[shop])
    return df


def plot_optimization_cost_history(history):
    f, ax = plt.subplots(figsize=(10,6))
    ax.plot(history)
    ax.set_yscale('log')
    ax.set_ylabel('Solution cost (log-scale)')
    ax.set_xlabel('Iteration')
    ax.set_title('Evolution of the solution cost along algorithm iterations')
    plt.show()
    return f
