#include <bits/stdc++.h>
#include <filesystem> 
using namespace std;

// ===== 全局参数（与 Python 一致） =====
static const int L = 500;
static const int N = 250;
static const double move_A_rate = 0.8;
static const double A_to_B_rate = 0.01;
static const double move_B_rate = 0.8;
static const double B_to_A_rate = 0.01;
static const int steps = 5000000;
static const int burning_steps = 1000000;

// 随机数
struct RNG {
    mt19937_64 eng;
    uniform_real_distribution<double> uni{0.0, 1.0};
    RNG() : eng(random_device{}()) {}
    inline double r() { return uni(eng); }
} rng;

// 写 CSV 工具
void write_csv(const std::string& filename, const std::vector<double>& rho) {
    namespace fs = std::filesystem;
    fs::create_directories("results");  // 确保 results 文件夹存在

    std::ofstream ofs("results/" + filename);  // 写入 results 文件夹
    ofs << "i,rho\n";
    for (int i = 0; i < (int)rho.size(); ++i) {
        ofs << (i + 1) << "," << rho[i] << "\n";
    }
}

// 单次模拟：返回 (rho_A, rho_B)
pair<vector<double>, vector<double>>
simulate_fully_parallel(double entry_rate, double exit_B_rate) {
    vector<int> A(L, 0), B(L, 0);

    // 初始化：从 2L 中无放回随机选 N 个位置
    vector<int> all(2 * L);
    iota(all.begin(), all.end(), 0);
    shuffle(all.begin(), all.end(), rng.eng);
    for (int k = 0; k < N; ++k) {
        int pos = all[k];
        if (pos < L) A[pos] = 1;
        else B[pos - L] = 1;
    }

    vector<long long> A_count(L, 0), B_count(L, 0);

    // 为了模拟“提案覆盖”，用可覆盖的记录结构
    // 我们把提案放到两个 map-like 的数组中：对每个源格（A/B,i）最多一个目标
    // 目标冲突用目标占用集合解决
    struct Move { char toLane; int toIdx; bool valid=false; };
    vector<Move> propA(L), propB(L);

    for (int t = 0; t < steps; ++t) {
        vector<int> A_next = A, B_next = B;

        // 清空提案
        for (int i = 0; i < L; ++i) { propA[i].valid = false; propB[i].valid = false; }

        // 内部移动 + 交换提案（顺序与 Python 保持：A内移、B内移、A->B、B->A；后面的可覆盖前面的同源提案）
        for (int i = 0; i < L; ++i) {
            // A 内部移动：i -> i+1
            if (i < L - 1 && A[i] == 1 && A[i+1] == 0 && rng.r() < move_A_rate) {
                propA[i] = Move{'A', i+1, true};
            }
            // B 内部移动：i -> i-1
            if (i > 0 && B[i] == 1 && B[i-1] == 0 && rng.r() < move_B_rate) {
                propB[i] = Move{'B', i-1, true};
            }
            // A -> B（覆盖 A 的同源提案）
            if (A[i] == 1 && B[i] == 0 && rng.r() < A_to_B_rate) {
                propA[i] = Move{'B', i, true};
            }
            // B -> A（覆盖 B 的同源提案）
            if (B[i] == 1 && A[i] == 0 && rng.r() < B_to_A_rate) {
                propB[i] = Move{'A', i, true};
            }
        }

        // 冲突检测：按 i 从小到大接受第一个占据某目标的提案
        // 使用两个目标占用数组，避免 pair hash
        vector<char> tgtA(L, 0), tgtB(L, 0);

        // 先处理 A 的提案
        for (int i = 0; i < L; ++i) if (propA[i].valid) {
            char toLane = propA[i].toLane; int j = propA[i].toIdx;
            bool ok = (toLane=='A') ? (!tgtA[j]) : (!tgtB[j]);
            if (ok) {
                // 应用到 next
                A_next[i] = 0;
                if (toLane=='A') { A_next[j] = 1; tgtA[j] = 1; }
                else             { B_next[j] = 1; tgtB[j] = 1; }
            }
        }
        // 再处理 B 的提案
        for (int i = 0; i < L; ++i) if (propB[i].valid) {
            char toLane = propB[i].toLane; int j = propB[i].toIdx;
            bool ok = (toLane=='A') ? (!tgtA[j]) : (!tgtB[j]);
            if (ok) {
                B_next[i] = 0;
                if (toLane=='A') { A_next[j] = 1; tgtA[j] = 1; }
                else             { B_next[j] = 1; tgtB[j] = 1; }
            }
        }

        // 入口与出口（注意用的是旧态 A[0], B[0] 判定，行为与原脚本一致）
        if (rng.r() < entry_rate && A[0] == 0) A_next[0] = 1;
        if (B[0] == 1 && rng.r() < exit_B_rate) B_next[0] = 0;

        // 更新
        A.swap(A_next);
        B.swap(B_next);

        // 统计密度
        if (t >= burning_steps) {
            for (int i = 0; i < L; ++i) {
                A_count[i] += A[i];
                B_count[i] += B[i];
            }
        }
    }

    int avg_steps = steps - burning_steps;
    vector<double> rhoA(L), rhoB(L);
    for (int i = 0; i < L; ++i) {
        rhoA[i] = double(A_count[i]) / double(avg_steps);
        rhoB[i] = double(B_count[i]) / double(avg_steps);
    }
    return {rhoA, rhoB};
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // ===== 配置 =====
    // (a) vary α, β=0.05
    {
        const vector<double> alphas = {0.3,0.4,0.5,0.6,0.7,0.8,0.9};
        double beta = 0.05;
        for (double alpha : alphas) {
            cerr << "Simulating (a) beta=0.05, alpha=" << alpha << "\n";
            auto [rhoA, rhoB] = simulate_fully_parallel(alpha, beta);
            // 写 CSV（Lane A/B）
            {
                ostringstream fn;
                fn << fixed << setprecision(2);
                fn << "case_a__laneA__param_alpha_" << alpha << ".csv";
                write_csv(fn.str(), rhoA);
            }
            {
                ostringstream fn;
                fn << fixed << setprecision(2);
                fn << "case_a__laneB__param_alpha_" << alpha << ".csv";
                write_csv(fn.str(), rhoB);
            }
        }
    }

    // (b) α=0.35, vary β
    {
        const double alpha = 0.35;
        const vector<double> betas = {0.20,0.30,0.32,0.34,0.36,0.38,0.40,0.50};
        for (double beta : betas) {
            cerr << "Simulating (b) alpha=0.35, beta=" << beta << "\n";
            auto [rhoA, rhoB] = simulate_fully_parallel(alpha, beta);
            {
                ostringstream fn;
                fn << fixed << setprecision(2);
                fn << "case_b__laneA__param_beta_" << beta << ".csv";
                write_csv(fn.str(), rhoA);
            }
            {
                ostringstream fn;
                fn << fixed << setprecision(2);
                fn << "case_b__laneB__param_beta_" << beta << ".csv";
                write_csv(fn.str(), rhoB);
            }
        }
    }

    // (c) α=0.7, vary β
    {
        const double alpha = 0.7;
        const vector<double> betas = {0.3,0.4,0.5,0.6,0.65,0.7,0.75,0.8,0.85,0.9};
        for (double beta : betas) {
            cerr << "Simulating (c) alpha=0.7, beta=" << beta << "\n";
            auto [rhoA, rhoB] = simulate_fully_parallel(alpha, beta);
            {
                ostringstream fn;
                fn << fixed << setprecision(2);
                fn << "case_c__laneA__param_beta_" << beta << ".csv";
                write_csv(fn.str(), rhoA);
            }
            {
                ostringstream fn;
                fn << fixed << setprecision(2);
                fn << "case_c__laneB__param_beta_" << beta << ".csv";
                write_csv(fn.str(), rhoB);
            }
        }
    }

    // (d) vary α, β=0.4
    {
        const vector<double> alphas = {0.0,0.15,0.30,0.45,0.50,0.53,0.56,0.60,0.75,0.90};
        const double beta = 0.4;
        for (double alpha : alphas) {
            cerr << "Simulating (d) beta=0.4, alpha=" << alpha << "\n";
            auto [rhoA, rhoB] = simulate_fully_parallel(alpha, beta);
            {
                ostringstream fn;
                fn << fixed << setprecision(2);
                fn << "case_d__laneA__param_alpha_" << alpha << ".csv";
                write_csv(fn.str(), rhoA);
            }
            {
                ostringstream fn;
                fn << fixed << setprecision(2);
                fn << "case_d__laneB__param_alpha_" << alpha << ".csv";
                write_csv(fn.str(), rhoB);
            }
        }
    }

    cerr << "All simulations done. CSV files written.\n";
    return 0;
}
