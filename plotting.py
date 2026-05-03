import matplotlib.pyplot as plt
import os

def plot_results(t_eval, sol, res_t, res_d, save_dir):
    plt.figure(figsize=(15, 6))
    for i in range(3):
        plt.subplot(2, 2, i+1)
        plt.plot(t_eval, sol.y[i], 'k-', alpha=0.2, lw=6, label='Real')
        plt.plot(t_eval, res_t[:, i], 'g-', lw=2, label=r'Match ($\beta = 0.3$)')
        plt.plot(t_eval, res_d[:, i], 'r--', lw=2, label=r'Drift ($\beta = 0.5$)')
        plt.title(['Susceptibles (S)', 'Infected (I)', 'Recovered (R)'][i])
        plt.grid(False)
    
    plt.subplot(2, 2, 4)
    plt.plot(t_eval, torch.sum(res_t, dim=1), 'g-')
    plt.plot(t_eval, torch.sum(res_d, dim=1), 'r--')
    plt.axhline(y=1.0, color='black', linestyle=':', alpha=0.5)
    plt.title('Sum (S+I+R)')
    plt.ylim(0.95, 1.05)
    plt.grid(False)
    
    plt.tight_layout()
    plt.legend()
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(f"{save_dir}/results.png")
    plt.close()
