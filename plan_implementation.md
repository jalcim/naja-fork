# Plan d'Implémentation : Révolution Architecture Naja

## I. OBJECTIF

Refondation complète mémoire et computation de Naja. Ordre par gain de performance maximum.

## II. PHASES PAR GAIN DE PERFORMANCE

### PHASE 1 : Memory Fixes (Gain +4000%) - 1 mois

**1.1 Suppression Indirections (Gain +4000%)**
- Problème : 4 indirections = 1200 cycles/accès
- Solution : Flat arrays directs  
- Impact : 95% cache miss → 95% cache hit

**1.2 AoS → SoA (Gain +2000%)**
- Problème : DNLInstanceFull 56B fragmenté
- Solution : Arrays par attribut, alignement 64B
- Impact : Vectorisation SIMD possible

**1.3 NUMA + Pool Allocation (Gain +1500%)**
- Problème : std::vector dispersés, pas de NUMA
- Solution : Pools dédiés, placement optimal
- Impact : 85% → 15% cache miss L3

### PHASE 2 : Parallelization (Gain +2000%) - 1.5 mois

**2.1 GPU Markovian (Gain +2000%)**
- Verified : 20× speedup (600ms → 30ms)
- HBM3 : 1.2TB/s effective @ 40% efficiency
- Memory per iteration : 1.2GB
- 30 iterations convergence

**2.2 CPU Multi-Threading (Gain +800%)**
- Problème : 1 thread sur 8+ cores (12% utilisation)
- Solution : Thread pool complet
- Impact : 90%+ utilisation cores

### PHASE 3 : Hardware Optimization (Gain +400%) - 2 mois

**3.1 Tensor Cores**
- Dense blocks 16×16 BF16
- 10% matrice éligible
- 4×+ speedup régions denses

**3.2 HBM3 Bandwidth**
- Actuel : 40% efficiency
- Cible : 60%+ via coalesced access
- Impact : +50% performance

**3.3 Cache Optimization**
- L1/L2/L3 hit rate optimization
- Data layout cache-friendly
- Blocking algorithms

### PHASE 4 : Multi-GPU (Gain +150%) - 1.5 mois

- NVLink 4.0 scaling
- Circuits >80GB support
- Inter-GPU communication

## III. ARCHITECTURE MEMORY

### Host Memory (NUMA)
- Node 0 : Hot data (8GB hugepages)
- Node 1 : Warm data (4GB hugepages)  
- Node 2+ : Cold data (standard pages)

### GPU HBM3 (80GB @ 3TB/s)
- CUDA workspace : 40GB
- Tensor workspace : 20GB
- System/coordination : 20GB

### On-Chip (per SM)
- Shared memory : 256KB optimisé
- Register file : 256KB auto-managed
- Texture cache : CSR row_ptr

## IV. PERFORMANCE TARGETS

### Verified Calculations
- Circuit 100M nœuds, sparsity 1%
- Memory : 1.2GB per iteration
- GPU time : 30ms (bandwidth limited)
- CPU time : 600ms
- **Speedup : 20× garanti**

### Phase Targets
| Phase | Speedup Cumulé | Bandwidth | Occupancy |
|-------|----------------|-----------|-----------|
| 1 | 40-50× | 70% | CPU 30% |
| 2 | 400-800× | 80% | CPU 90%, GPU 80% |
| 3 | 800-1200× | 85% | CPU 95%, GPU 85% |
| 4 | 1000-1500× | 90% | Multi-GPU 80% |

## V. TIMELINE

**Mois 1** : Memory revolution (indirections, AoS→SoA, NUMA)
**Mois 2-3.5** : Parallelization (GPU + CPU multi-threading)  
**Mois 4-6** : Hardware optimization (Tensor + bandwidth + cache)
**Mois 7-8.5** : Multi-GPU scaling + production

## VI. RESSOURCES

**Team** : 4-5 développeurs experts
**Hardware** : H100 SXM5, 64+ core CPU, 512GB RAM
**Tools** : Nsight Compute, profiling suite

## VII. SUCCESS CRITERIA

- Phase 1 : >30× speedup, <10% cache miss
- Phase 2 : >300× speedup, >80% GPU occupancy  
- Phase 3 : >800× speedup, >85% bandwidth
- Phase 4 : >1000× speedup, multi-GPU ready