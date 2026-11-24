#include <map>
#include <set>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdio>
#include "TFile.h"
#include "TDirectory.h"
#include "TH1D.h"
#include "TGraph.h"
#include "TString.h"
#include "TRandom3.h"
#include "TKey.h"

// usage: root -l -q 'thresholdTimes.C("in/modules.root","out.root","_px1-1",110,0.5,1.5,0)'
// args: inFile, outFile, pixTag, noise_sigma [e], noise_dt [ns], tmax [ns], legacy (1 = old discrete sum)
//   noise_sigma = 0   -> only the noise-free crossing is computed
//   noise_dt    > 0   -> one noise value per absolute-time window (seeded by event+window)
//   noise_dt    = 0   -> new value every bin (old behavior, timestep dependent)
// Writes out.root (hThresholdTime = noisy if noise on, else signal-only; hSignalOnly) and out.csv

namespace {
const double kThr = -350.0, kRise = 0.5, kInt = 10.0;   // unchanged settings

// integral of h(s) = (s/kRise) exp(-s/kInt) from 0 to s
double Hint(double s) {
    if (s <= 0) return 0;
    return (kInt*kInt - kInt*std::exp(-s/kInt)*(s + kInt)) / kRise;
}

std::vector<double> shapeSignal(const double* t, const double* q, int n, bool legacy) {
    std::vector<double> s(n, 0.0);
    for (int j = 0; j < n; ++j) {
        for (int k = 0; k <= j; ++k) {
            if (legacy) {
                double dQ = (k == 0) ? q[0] : q[k] - q[k-1];
                double dt = t[j] - t[k];
                if (dt >= 0) s[j] += dQ * (dt/kRise) * std::exp(-dt/kInt);
            } else if (k == 0) {
                double dt = t[j] - t[0];
                s[j] += q[0] * (dt/kRise) * std::exp(-dt/kInt);
            } else {
                double w = t[k] - t[k-1];
                if (w <= 0) continue;
                // charge q[k]-q[k-1] assumed to arrive uniformly over [t[k-1], t[k]]
                s[j] += (q[k] - q[k-1]) / w * (Hint(t[j]-t[k-1]) - Hint(t[j]-t[k]));
            }
        }
    }
    return s;
}

std::vector<double> makeNoise(const double* t, int n, int ev, double sigma, double ndt) {
    std::vector<double> z(n, 0.0);
    if (sigma <= 0) return z;
    if (ndt > 0) {
        int nw = (int)std::floor(std::max(0.0, t[n-1]) / ndt) + 1;
        std::vector<double> w(nw);
        for (int i = 0; i < nw; ++i) {
            TRandom3 r((UInt_t)(1000003UL * (unsigned long)(ev + 1) + (unsigned long)i + 1));
            w[i] = r.Gaus(0, sigma);
        }
        for (int j = 0; j < n; ++j) {
            int idx = (int)std::floor(std::max(0.0, t[j]) / ndt);
            z[j] = w[std::min(idx, nw - 1)];
        }
    } else {
        TRandom3 r((UInt_t)(ev + 1));
        for (int j = 0; j < n; ++j) z[j] = r.Gaus(0, sigma);
    }
    return z;
}

double firstCrossing(const double* t, const std::vector<double>& s, int n) {
    if (n > 0 && s[0] <= kThr) return t[0];
    for (int i = 1; i < n; ++i) {
        if (s[i-1] > kThr && s[i] <= kThr) {
            double frac = (kThr - s[i-1]) / (s[i] - s[i-1]);
            return t[i-1] + frac * (t[i] - t[i-1]);
        }
    }
    return -1.0;
}

void report(const char* label, std::vector<double> v, int nEvents) {
    if (v.size() < 10) { std::printf("%-14s only %zu crossings\n", label, v.size()); return; }
    std::sort(v.begin(), v.end());
    double sum = 0; for (double x : v) sum += x;
    double mean = sum / v.size(), var = 0;
    for (double x : v) var += (x - mean) * (x - mean);
    double sd = std::sqrt(var / (v.size() - 1)), se = sd / std::sqrt((double)v.size());
    double lo = v[(size_t)(0.16 * (v.size() - 1))], hi = v[(size_t)(0.84 * (v.size() - 1))];
    int late = 0; for (double x : v) if (x > 1.0) ++late;
    std::printf("%-14s N=%zu/%d  mean=%.1f +- %.1f ps  median=%.1f  sd=%.1f  half16-84=%.1f ps  (>1ns: %d)\n",
        label, v.size(), nEvents, 1e3*mean, 1e3*se, 1e3*v[v.size()/2], 1e3*sd, 1e3*0.5*(hi-lo), late);
}
} // namespace

void thresholdTimes(const char* inFile = "modules.root",
                    const char* outFile = "threshold_times.root",
                    const char* pixTag = "_px1-1",
                    double noise_sigma = 110.0,
                    double noise_dt = 0.5,
                    double tmax = 1.5,
                    int legacy = 0)
{
    TFile* f = TFile::Open(inFile);
    if (!f || f->IsZombie()) { std::cout << "ERROR: cannot open " << inFile << "\n"; return; }
    TDirectory* dir = (TDirectory*)f->Get("PulseTransfer/detector1");
    if (!dir) { std::cout << "ERROR: directory not found\n"; return; }

    struct Rec { double ts = -1, tn = -1, qEnd = 0, sMin = 0; };
    std::map<int, Rec> recs;
    int nGraphs = 0;
    bool printedDiag = false;

    TIter next(dir->GetListOfKeys());
    TKey* key;
    while ((key = (TKey*)next())) {
        TString name = key->GetName();
        if (!name.BeginsWith("charge_ev") || !name.Contains(pixTag)) continue;
        int evID = 0;
        std::sscanf(name.Data(), "charge_ev%d", &evID);
        TGraph* gr = (TGraph*)key->ReadObj();
        if (!gr) continue;
        ++nGraphs;
        gr->Sort();
        int nAll = gr->GetN(), n = 0;
        const double* t = gr->GetX();
        const double* q = gr->GetY();
        while (n < nAll && t[n] <= tmax) ++n;
        if (n < 2) { delete gr; continue; }

        if (!printedDiag) {
            std::printf("DIAG %s: points=%d  t[0]=%.5f  t[1]-t[0]=%.5f  t[last]=%.4f ns  q[0]=%.1f  q[last]=%.1f  q_extreme=%.1f e\n",
                name.Data(), nAll, t[0], t[1]-t[0], t[nAll-1], q[0], q[nAll-1],
                *std::max_element(q, q+nAll, [](double a, double b){return std::fabs(a) < std::fabs(b);}));
            printedDiag = true;
        }

        std::vector<double> s = shapeSignal(t, q, n, legacy != 0);
        double tsig = firstCrossing(t, s, n);
        double sMin = *std::min_element(s.begin(), s.end());
        double tnoi = -1;
        if (noise_sigma > 0) {
            std::vector<double> z = makeNoise(t, n, evID, noise_sigma, noise_dt);
            std::vector<double> sn(n);
            for (int j = 0; j < n; ++j) sn[j] = s[j] + z[j];
            tnoi = firstCrossing(t, sn, n);
        }
        Rec& r = recs[evID];
        if (tsig >= 0 && (r.ts < 0 || tsig < r.ts)) r.ts = tsig;
        if (tnoi >= 0 && (r.tn < 0 || tnoi < r.tn)) r.tn = tnoi;
        r.qEnd = q[nAll-1];
        r.sMin = std::min(r.sMin, sMin);
        delete gr;
    }

    TH1D* hSig = new TH1D("hSignalOnly", "Noise-free crossing;time [ns];events", 750, 0, 1.5);
    TH1D* hNoi = new TH1D("hNoisy", "Noisy crossing;time [ns];events", 750, 0, 1.5);
    hSig->SetDirectory(0); hNoi->SetDirectory(0);
    std::vector<double> vs, vn, diff;
    int early = 0;
    TString csvName(outFile); csvName.ReplaceAll(".root", ".csv");
    std::ofstream csv(csvName.Data());
    csv << "event,t_signal_ns,t_noisy_ns,q_end_e,min_shaped_e\n";
    for (auto const& kv : recs) {
        const Rec& r = kv.second;
        csv << kv.first << "," << r.ts << "," << r.tn << "," << r.qEnd << "," << r.sMin << "\n";
        if (r.ts >= 0) { vs.push_back(r.ts); hSig->Fill(r.ts); }
        if (r.tn >= 0) { vn.push_back(r.tn); hNoi->Fill(r.tn); }
        if (r.ts >= 0 && r.tn >= 0) { diff.push_back(r.tn - r.ts); if (r.tn < r.ts - 0.05) ++early; }
    }
    csv.close();

    std::printf("File: %s  method=%s  noise_sigma=%.0f e  noise_dt=%.3f ns  tmax=%.2f ns\n",
        inFile, legacy ? "LEGACY" : "exact", noise_sigma, noise_dt, tmax);
    std::printf("Graphs processed: %d   events with central graph: %zu (compare number_of_events)\n", nGraphs, recs.size());
    report("signal-only", vs, (int)recs.size());
    if (noise_sigma > 0) {
        report("with noise", vn, (int)recs.size());
        if (diff.size() > 10) {
            double m = 0; for (double d : diff) m += d; m /= diff.size();
            std::printf("noisy - signal: mean %.1f ps over %zu events; noise triggers >50 ps early: %d\n",
                1e3*m, diff.size(), early);
        }
    }

    TFile out(outFile, "RECREATE");
    TH1D* hMain = (noise_sigma > 0) ? hNoi : hSig;
    hMain->SetName("hThresholdTime");
    hMain->Write();
    hSig->SetName("hSignalOnly"); hSig->Write();
    out.Close();
    std::cout << "DONE -> " << outFile << " and " << csvName << "\n";
}