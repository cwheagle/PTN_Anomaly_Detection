<template>
  <div class="space-y-8 text-slate-200">
    <div class="bg-slate-800 p-10 rounded-2xl border border-slate-700 shadow-xl">
      <div class="flex justify-between items-center mb-10">
        <div class="flex items-center gap-6">
          <div class="p-4 bg-blue-500/10 rounded-xl">
            <svg xmlns="http://www.w3.org/2000/svg" class="w-10 h-10 text-blue-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"></path></svg>
          </div>
          <div>
            <h2 class="text-3xl font-bold text-slate-100">Model Management</h2>
            <p class="text-lg text-slate-400">Configure, train, and monitor anomaly detection models.</p>
          </div>
        </div>
        <button @click="store.fetchModelStatus()" 
                :disabled="store.isRefreshingModelStatus"
                class="p-3 text-slate-400 hover:text-blue-400 transition-colors">
          <svg xmlns="http://www.w3.org/2000/svg" :class="['w-7 h-7', store.isRefreshingModelStatus ? 'animate-spin' : '']" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"></polyline><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path></svg>
        </button>
      </div>

      <!-- Data Drift Monitor -->
      <div class="mb-10 p-6 bg-slate-900/50 rounded-2xl border border-slate-700/50 shadow-inner">
        <div class="flex justify-between items-center mb-6">
          <div class="flex items-center gap-3">
            <div class="p-2 bg-purple-500/10 rounded-lg">
              <svg xmlns="http://www.w3.org/2000/svg" class="w-6 h-6 text-purple-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"></path></svg>
            </div>
            <div>
              <h3 class="text-lg font-bold text-slate-200">Data Drift Monitor</h3>
              <p class="text-xs text-slate-400">포트별 24H 평균 점수의 분포를 학습 시 기준 분포와 비교합니다. 다수 포트가 함께 이동하면(광역) 후보 모델 학습, 소수 포트만 급등하면(국소) 장애 의심으로 보고 재학습하지 않습니다.</p>
            </div>
          </div>
          <button @click="handleCheckDrift"
                  :disabled="store.isCheckingDrift"
                  class="px-5 py-2.5 bg-purple-600/20 hover:bg-purple-600/30 border border-purple-500/30 text-purple-400 rounded-xl text-sm font-bold transition-all flex items-center gap-2">
            <svg v-if="store.isCheckingDrift" class="animate-spin h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
            Check Drift Now
          </button>
        </div>
        
        <div class="grid grid-cols-2 gap-6" v-if="store.driftStatus && !store.driftStatus.message">
          <div v-for="ft in ['traffic', 'optical']" :key="ft" class="p-4 bg-slate-800 rounded-xl border border-slate-700">
            <div class="flex justify-between items-center mb-4">
              <span class="uppercase font-bold text-xs tracking-wider text-slate-400">{{ ft }} Track</span>
              <span :class="['px-2 py-0.5 rounded text-[10px] font-bold uppercase', driftBadge(ft as string).cls]">
                {{ driftBadge(ft as string).text }}
              </span>
            </div>
            <div class="flex justify-between items-end">
              <div>
                <div class="text-xs text-slate-500 mb-1">Median Ratio (포트 중앙값 / 기준)</div>
                <div class="text-2xl font-mono text-slate-200 font-bold">
                  {{ (store.driftStatus[ft]?.median_ratio ?? store.driftStatus[ft]?.drift_ratio)?.toFixed(2) }}<span class="text-sm text-slate-500">x</span>
                </div>
              </div>
              <div class="text-right">
                <div class="text-[10px] text-slate-500 mb-1">Drifted Ports (기준 p90 초과)</div>
                <div class="font-mono text-xs text-slate-400">
                  {{ store.driftStatus[ft]?.drifted_port_fraction !== undefined ? (store.driftStatus[ft].drifted_port_fraction * 100).toFixed(1) + '%' : '-' }}
                  <span class="text-slate-600">/ {{ store.driftStatus[ft]?.ports ?? '-' }} ports</span>
                </div>
                <div class="text-[10px] text-slate-500 mb-1 mt-1">Baseline / 24H Average MSE</div>
                <div class="font-mono text-xs text-slate-400">{{ store.driftStatus[ft]?.baseline_mse?.toFixed(5) }} / {{ store.driftStatus[ft]?.mean_mse?.toFixed(5) }}</div>
              </div>
            </div>
            <!-- 재학습 결정: 지속성(연속 일수) · 쿨다운 · 사유 -->
            <div class="mt-4 pt-3 border-t border-slate-700/60 text-[11px] space-y-1">
              <div class="flex justify-between">
                <span class="text-slate-500">Decision</span>
                <span :class="['font-bold uppercase', store.driftStatus.retrain?.[ft]?.action === 'train' ? 'text-amber-400' : 'text-slate-300']">{{ store.driftStatus.retrain?.[ft]?.action || '-' }}</span>
              </div>
              <div class="text-slate-400 leading-relaxed">{{ store.driftStatus.retrain?.[ft]?.reason || store.driftStatus[ft]?.message || '' }}</div>
              <div v-if="store.retrainState[ft]" class="flex justify-between text-slate-500">
                <span>Persistence {{ store.retrainState[ft].streak_days }}/{{ store.retrainState[ft].persist_required }} days · mode {{ store.retrainState[ft].mode }}</span>
                <span v-if="store.retrainState[ft].cooldown_remaining_hours > 0">Cooldown {{ store.retrainState[ft].cooldown_remaining_hours }}h left</span>
              </div>
              <div v-if="store.driftStatus[ft]?.kind === 'localized' && store.driftStatus[ft]?.top_ports?.length" class="text-amber-400/80">
                Top ports: {{ store.driftStatus[ft].top_ports.slice(0, 3).map((p: any) => `${p.ip_addr}/${p.cid}/${p.lid}`).join(', ') }}
              </div>
            </div>
          </div>
        </div>
        <div v-else-if="store.driftStatus && store.driftStatus.message" class="text-center p-6 text-sm text-slate-400 italic bg-slate-800 rounded-xl border border-slate-700">
          {{ store.driftStatus.message }}
        </div>
        <div v-else class="text-center p-6 text-sm text-slate-500 italic bg-slate-800 rounded-xl border border-slate-700">
          No drift data available. Click "Check Drift Now" or wait for the scheduled check.
        </div>
        
        <div v-if="store.driftStatus?.auto_retrain_triggered" class="mt-4 p-3 bg-amber-500/10 border border-amber-500/30 rounded-lg flex items-center gap-3">
          <span class="w-2 h-2 rounded-full bg-amber-400 animate-ping"></span>
          <span class="text-sm text-amber-400 font-bold">광역 드리프트가 지속되어 후보 모델 학습을 시작했습니다. 완료 후 Model Versions 에서 Gate 결과를 확인하고 Promote 하세요.</span>
        </div>
      </div>

      <!-- Main Layout: 2 Columns (Traffic | Optical) -->
      <div class="grid grid-cols-1 xl:grid-cols-2 gap-10">
        <div v-for="(info, ft) in store.modelStatus" :key="ft" 
             class="bg-slate-900/40 rounded-2xl border border-slate-700 overflow-hidden flex flex-col">
          
          <!-- Header -->
          <div class="p-6 border-b border-slate-700 bg-slate-800/50 flex justify-between items-center">
            <div class="flex items-center gap-4">
              <h3 class="font-bold text-blue-400 uppercase tracking-widest text-base">{{ ft }} Specialist Model</h3>
              <span :class="['px-3 py-1 rounded text-xs font-bold uppercase', 
                            info.exists ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30' : 'bg-rose-500/20 text-rose-400 border border-rose-500/30']">
                {{ info.exists ? 'Active' : 'Missing' }}
              </span>
              <span v-if="info.active_version" class="px-2 py-1 rounded text-xs font-mono text-slate-300 bg-slate-700/50 border border-slate-600">{{ info.active_version }}</span>
            </div>
            <div class="text-sm">
              <span class="text-slate-500">Last Trained:</span>
              <span class="text-slate-300 font-mono ml-2">{{ info.last_trained ? info.last_trained.split(' ')[0] : 'Never' }}</span>
            </div>
          </div>

          <div class="p-8 space-y-12">
            <!-- 1. Training Configuration (Top) -->
            <div class="space-y-8">
              <div class="flex items-center justify-between">
                <div class="flex items-center gap-3">
                  <div class="w-2 h-5 bg-amber-500 rounded-full"></div>
                  <h4 class="text-sm font-bold text-slate-200 uppercase tracking-wider">Training Configuration</h4>
                </div>
                <div class="text-xs text-slate-500">
                  Samples: <span class="text-slate-300 font-mono text-sm">{{ info.samples_used.toLocaleString() }}</span>
                </div>
              </div>

              <div class="space-y-6 bg-slate-800/30 p-6 rounded-xl border border-slate-700/50">
                <div class="grid grid-cols-2 gap-x-8 gap-y-6">
                  <div class="space-y-2">
                    <div class="group relative flex items-center gap-2">
                      <label class="text-xs text-slate-500 uppercase font-bold block">Epochs</label>
                      <span class="text-slate-700 cursor-help text-xs">ⓘ</span>
                      <div class="absolute bottom-full left-0 mb-2 invisible group-hover:visible bg-slate-700 text-white text-[10px] px-2 py-1 rounded whitespace-nowrap z-10 shadow-xl border border-slate-600">
                        Range: 1 ~ 1000
                      </div>
                    </div>
                    <input type="number" v-model.number="trainConfigs[ft as string].epochs" 
                           min="1" max="1000"
                           class="w-full bg-slate-900 border border-slate-700 rounded-lg p-3 text-sm text-slate-200 focus:border-amber-500/50 outline-none transition-colors" />
                  </div>

                  <div class="space-y-2">
                    <div class="group relative flex items-center gap-2">
                      <label class="text-xs text-slate-500 uppercase font-bold block">Learning Rate</label>
                      <span class="text-slate-700 cursor-help text-xs">ⓘ</span>
                      <div class="absolute bottom-full left-0 mb-2 invisible group-hover:visible bg-slate-700 text-white text-[10px] px-2 py-1 rounded whitespace-nowrap z-10 shadow-xl border border-slate-600">
                        Range: 0.0001 ~ 0.1
                      </div>
                    </div>
                    <input type="number" step="0.0001" v-model.number="trainConfigs[ft as string].learning_rate" 
                           min="0.0001" max="0.1"
                           class="w-full bg-slate-900 border border-slate-700 rounded-lg p-3 text-sm text-slate-200 focus:border-amber-500/50 outline-none transition-colors" />
                  </div>

                  <div class="space-y-2">
                    <div class="group relative flex items-center gap-2">
                      <label class="text-xs text-slate-500 uppercase font-bold block">Batch Size</label>
                      <span class="text-slate-700 cursor-help text-xs">ⓘ</span>
                      <div class="absolute bottom-full left-0 mb-2 invisible group-hover:visible bg-slate-700 text-white text-[10px] px-2 py-1 rounded whitespace-nowrap z-10 shadow-xl border border-slate-600">
                        Range: 1 ~ 1024
                      </div>
                    </div>
                    <input type="number" v-model.number="trainConfigs[ft as string].batch_size" 
                           min="1" max="1024"
                           class="w-full bg-slate-900 border border-slate-700 rounded-lg p-3 text-sm text-slate-200 focus:border-amber-500/50 outline-none transition-colors" />
                  </div>

                  <div class="space-y-2">
                    <div class="group relative flex items-center gap-2">
                      <label class="text-xs text-slate-500 uppercase font-bold block">Percentile</label>
                      <span class="text-slate-700 cursor-help text-xs">ⓘ</span>
                      <div class="absolute bottom-full left-0 mb-2 invisible group-hover:visible bg-slate-700 text-white text-[10px] px-2 py-1 rounded whitespace-nowrap z-10 shadow-xl border border-slate-600">
                        Range: 90.0 ~ 99.9
                      </div>
                    </div>
                    <input type="number" step="0.1" v-model.number="trainConfigs[ft as string].threshold_percentile" 
                           min="90" max="99.9"
                           class="w-full bg-slate-900 border border-slate-700 rounded-lg p-3 text-sm text-slate-200 focus:border-amber-500/50 outline-none transition-colors" />
                  </div>

                  <div class="space-y-2">
                    <div class="group relative flex items-center gap-2">
                      <label class="text-xs text-slate-500 uppercase font-bold block">Early Stop Patience</label>
                      <span class="text-slate-700 cursor-help text-xs">ⓘ</span>
                      <div class="absolute bottom-full left-0 mb-2 invisible group-hover:visible bg-slate-700 text-white text-[10px] px-2 py-1 rounded whitespace-nowrap z-10 shadow-xl border border-slate-600">
                        Stop after N epochs with no improvement.
                      </div>
                    </div>
                    <input type="number" v-model.number="trainConfigs[ft as string].patience" 
                           min="1" max="100"
                           class="w-full bg-slate-900 border border-slate-700 rounded-lg p-3 text-sm text-slate-200 focus:border-amber-500/50 outline-none transition-colors" />
                  </div>
                </div>

                <div class="space-y-4 pt-4">
                  <div class="flex items-center gap-3">
                    <div class="w-1.5 h-4 bg-amber-500/50 rounded-full"></div>
                    <span class="text-xs font-bold text-slate-400 uppercase tracking-wider">Dataset Range</span>
                  </div>
                  
                  <label class="flex items-start gap-3 text-xs text-slate-300 cursor-pointer">
                    <input type="checkbox" v-model="useRecentWindow[ft as string]" class="mt-0.5" />
                    <span>최근 구간 자동 (권장) — 최근 28일을 <b>포트 단위 홀드아웃</b>으로 학습/검증에 나눕니다. 날짜를 직접 지정하려면 해제하세요.</span>
                  </label>
                  <label class="flex items-start gap-3 text-xs text-slate-300 cursor-pointer">
                    <input type="checkbox" v-model="excludeSuspect[ft as string]" class="mt-0.5" />
                    <span>장애 의심 구간 자동 제외 (자기 알람 이력 + 지속 오류/광 하락/트래픽 급감 규칙). 의심 구간이 20% 를 넘으면 학습을 중단합니다.</span>
                  </label>
                  <div class="flex items-center gap-3 text-xs text-slate-300">
                    <span class="font-bold text-slate-400 uppercase tracking-wider">Alert Policy</span>
                    <select v-model="alertPolicy[ft as string]" class="bg-slate-800 border border-slate-700 rounded-lg p-1.5 text-xs text-slate-200">
                      <option value="">활성 모델의 정책 승계 (기본)</option>
                      <option value="default">default — 기본 정책</option>
                      <option value="precision">precision — 오탐 억제형 (조기 탐지 감소 대가, 모델과 짝으로 배포)</option>
                    </select>
                  </div>
                  <div :class="['grid grid-cols-2 gap-6 bg-slate-900/50 p-4 rounded-xl border border-slate-700/50', useRecentWindow[ft as string] ? 'opacity-40 pointer-events-none' : '']">
                    <div class="space-y-3">
                      <span class="text-xs font-bold text-blue-400 uppercase block border-b border-blue-500/20 pb-1">Training</span>
                      <div class="space-y-2">
                        <label class="text-[10px] text-slate-500 uppercase font-bold">Start Date</label>
                        <input type="date" v-model="dateConfigs[ft as string].train_start" class="w-full bg-slate-800 border border-slate-700 rounded-lg p-2 text-xs text-slate-200" />
                        <label class="text-[10px] text-slate-500 uppercase font-bold">End Date</label>
                        <input type="date" v-model="dateConfigs[ft as string].train_end" class="w-full bg-slate-800 border border-slate-700 rounded-lg p-2 text-xs text-slate-200" />
                      </div>
                    </div>
                    <div class="space-y-3">
                      <span class="text-xs font-bold text-emerald-400 uppercase block border-b border-emerald-500/20 pb-1">Validation</span>
                      <div class="space-y-2">
                        <label class="text-[10px] text-slate-500 uppercase font-bold">Start Date</label>
                        <input type="date" v-model="dateConfigs[ft as string].test_start" class="w-full bg-slate-800 border border-slate-700 rounded-lg p-2 text-xs text-slate-200" />
                        <label class="text-[10px] text-slate-500 uppercase font-bold">End Date</label>
                        <input type="date" v-model="dateConfigs[ft as string].test_end" class="w-full bg-slate-800 border border-slate-700 rounded-lg p-2 text-xs text-slate-200" />
                      </div>
                    </div>
                  </div>
                </div>

                <!-- Training Control (Inline Layout) -->
                <div v-if="isTraining[ft as string]" class="flex gap-2 w-full">
                  <div class="flex-1 flex items-center justify-between px-6 py-4 bg-amber-600/10 border border-amber-500/30 text-amber-500 rounded-xl font-bold opacity-90">
                    <div class="flex items-center gap-3">
                      <svg class="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                      <span class="text-sm">Training...</span>
                    </div>
                    <div v-if="store.modelStatus[ft]?.training" class="text-[11px] font-mono text-amber-400/90 bg-amber-500/10 px-3 py-1 rounded-lg border border-amber-500/20">
                      {{ store.modelStatus[ft].training.current_epoch }}/{{ store.modelStatus[ft].training.total_epochs }} 
                      | Loss: {{ store.modelStatus[ft].training.loss?.toFixed(5) }}
                      <span v-if="store.modelStatus[ft].training.val_loss !== null">
                        | Val: {{ store.modelStatus[ft].training.val_loss?.toFixed(5) }}
                      </span>
                    </div>
                  </div>
                  <button @click="handleStop(ft as string)"
                          class="px-5 bg-rose-500/10 hover:bg-rose-500/20 border border-rose-500/30 text-rose-500 rounded-xl text-xs font-bold transition-all flex items-center justify-center">
                    Stop
                  </button>
                </div>
                <button v-else
                        @click="handleTrain(ft as string)" 
                        class="w-full py-4 bg-amber-600/10 hover:bg-amber-600/20 border border-amber-500/30 text-amber-500 rounded-xl text-sm font-bold transition-all flex items-center justify-center gap-3">
                  Train
                </button>

                <!-- Error Message Display -->
                <div v-if="!isTraining[ft as string] && store.modelStatus[ft]?.training?.last_error" 
                     class="mt-4 p-4 bg-rose-500/10 border border-rose-500/30 rounded-xl flex items-start gap-3">
                  <svg xmlns="http://www.w3.org/2000/svg" class="w-5 h-5 text-rose-400 shrink-0 mt-0.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>
                  <div class="space-y-1">
                    <p class="text-xs font-bold text-rose-400 uppercase tracking-wider">Training Fail</p>
                    <p class="text-sm text-rose-300/80 leading-relaxed">{{ store.modelStatus[ft].training.last_error }}</p>
                  </div>
                </div>

                <!-- Success Message Display -->
                <div v-if="!isTraining[ft as string] && store.modelStatus[ft]?.training?.success_msg" 
                     class="mt-4 p-4 bg-emerald-500/10 border border-emerald-500/30 rounded-xl flex items-start gap-3">
                  <svg xmlns="http://www.w3.org/2000/svg" class="w-5 h-5 text-emerald-400 shrink-0 mt-0.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>
                  <div class="space-y-1">
                    <p class="text-xs font-bold text-emerald-400 uppercase tracking-wider">Training Success</p>
                    <p class="text-sm text-emerald-300/80 leading-relaxed">{{ store.modelStatus[ft].training.success_msg }}</p>
                  </div>
                </div>
              </div>
            </div>

            <!-- 2. Model Versions: 재학습 결과는 후보로만 저장되며, 승격(Promote)해야 실시간 엔진에 반영됨 -->
            <div class="space-y-4">
              <div class="flex items-center justify-between">
                <div class="flex items-center gap-3">
                  <div class="w-2 h-5 bg-blue-500 rounded-full"></div>
                  <h4 class="text-sm font-bold text-slate-200 uppercase tracking-wider">Model Versions</h4>
                </div>
                <button @click="handleRollback(ft as string)" :disabled="!canRollback(ft as string)"
                        class="px-3 py-1.5 text-xs font-bold rounded-lg border border-slate-600 text-slate-300 hover:bg-slate-700 disabled:opacity-30 disabled:cursor-not-allowed transition-colors">
                  Rollback
                </button>
              </div>
              <p class="text-xs text-slate-500 leading-relaxed">
                재학습 결과는 <span class="text-amber-400 font-bold">Candidate</span> 로만 저장됩니다. 승격 게이트(Gate) 결과와 임계치·검증 손실을 확인한 뒤 <span class="text-blue-400 font-bold">Promote</span> 해야 실시간 엔진에 반영됩니다. FAIL 은 경고를 확인하고 강제 승격할 수 있으나, G1(깨진 모델)은 승격할 수 없습니다.
              </p>
              <div class="overflow-x-auto bg-slate-800/30 rounded-xl border border-slate-700/50">
                <table class="w-full text-left text-xs">
                  <thead class="text-slate-500 uppercase text-[10px] tracking-wider">
                    <tr>
                      <th class="px-3 py-2">Version</th>
                      <th class="px-3 py-2">Trained</th>
                      <th class="px-3 py-2">Threshold</th>
                      <th class="px-3 py-2">Val Loss</th>
                      <th class="px-3 py-2">Samples</th>
                      <th class="px-3 py-2">Trigger</th>
                      <th class="px-3 py-2">Policy</th>
                      <th class="px-3 py-2">Gate</th>
                      <th class="px-3 py-2">Status</th>
                      <th class="px-3 py-2 text-right"></th>
                    </tr>
                  </thead>
                  <tbody class="divide-y divide-slate-700/50">
                    <template v-for="ver in versionsOf(ft as string)" :key="ver.version">
                    <tr class="hover:bg-slate-700/20">
                      <td class="px-3 py-2 font-mono font-bold text-slate-200">{{ ver.version }}</td>
                      <td class="px-3 py-2 font-mono text-slate-400">{{ ver.trained_at || '-' }}</td>
                      <td class="px-3 py-2 font-mono text-slate-300">{{ fmtNum(ver.threshold) }}</td>
                      <td class="px-3 py-2 font-mono text-slate-300">{{ fmtNum(ver.final_val_loss) }}</td>
                      <td class="px-3 py-2 font-mono text-slate-400">{{ ver.samples_used ? ver.samples_used.toLocaleString() : '-' }}</td>
                      <td class="px-3 py-2 text-slate-400">{{ ver.trigger || '-' }}<span v-if="ver.derived_from" class="text-slate-500"> ← {{ ver.derived_from }}</span></td>
                      <td class="px-3 py-2 text-slate-400">{{ ver.alert_policy?.preset || 'default (메타 없음)' }}</td>
                      <td class="px-3 py-2">
                        <button v-if="ver.gate" @click="toggleGate(ft as string, ver.version)"
                                :class="['px-2 py-0.5 rounded text-[10px] font-bold uppercase border', gateClass(ver.gate_status)]">{{ ver.gate_status }}</button>
                        <span v-else class="text-slate-600">—</span>
                      </td>
                      <td class="px-3 py-2">
                        <span :class="['px-2 py-0.5 rounded text-[10px] font-bold uppercase',
                                       ver.status === 'active' ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30' :
                                       ver.status === 'candidate' ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30' :
                                       'bg-slate-500/20 text-slate-400 border border-slate-500/30']">{{ ver.status }}</span>
                      </td>
                      <td class="px-3 py-2 text-right whitespace-nowrap">
                        <button v-if="ver.status !== 'active'" @click="handleRerunGate(ft as string, ver.version)"
                                class="mr-2 px-2 py-1 text-[11px] font-bold rounded-lg border border-slate-600 text-slate-300 hover:bg-slate-700 transition-colors">
                          Gate
                        </button>
                        <button v-if="ver.status !== 'active'" @click="handlePromote(ft as string, ver.version)"
                                class="px-3 py-1 text-[11px] font-bold rounded-lg bg-blue-600/20 border border-blue-500/30 text-blue-300 hover:bg-blue-600/30 transition-colors">
                          Promote
                        </button>
                      </td>
                    </tr>
                    <!-- 게이트 검사별 값 / 한계 / 메시지 -->
                    <tr v-if="ver.gate && isGateOpen(ft as string, ver.version)" class="bg-slate-900/50">
                      <td colspan="10" class="px-4 py-3">
                        <div class="text-[10px] text-slate-500 mb-2">
                          evaluated {{ ver.gate.evaluated_at }} · vs {{ ver.gate.data?.against || 'no active model' }}
                          <span v-if="ver.gate.data?.alert_policy"> · policy {{ ver.gate.data.alert_policy.candidate?.preset }} vs {{ ver.gate.data.alert_policy.active?.preset || '-' }}</span>
                          <span v-if="ver.suspect_stats"> · suspect {{ (ver.suspect_stats.fraction * 100).toFixed(1) }}% of training rows</span>
                        </div>
                        <div v-for="c in ver.gate.checks" :key="c.id" class="flex items-start gap-3 py-0.5">
                          <span :class="['px-1.5 py-0.5 rounded text-[10px] font-bold border w-14 text-center', gateClass(c.status)]">{{ c.id }} {{ c.status }}</span>
                          <span class="text-slate-300">{{ c.message }}</span>
                          <span v-if="c.value !== null && c.value !== undefined" class="ml-auto font-mono text-slate-500">{{ fmtNum(c.value) }}<span v-if="c.limit !== null && c.limit !== undefined"> / {{ fmtNum(c.limit) }}</span></span>
                        </div>
                      </td>
                    </tr>
                    </template>
                    <tr v-if="versionsOf(ft as string).length === 0">
                      <td colspan="10" class="px-3 py-6 text-center text-slate-500 italic">No versions yet.</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>


          </div>
        </div>
      </div>
    </div>

    <!-- Notification -->
    <div class="fixed bottom-10 left-1/2 -translate-x-1/2 z-50 transition-all duration-500" 
         :class="[notification.show ? 'translate-y-0 opacity-100' : 'translate-y-12 opacity-0 pointer-events-none']">
      <div :class="['px-8 py-4 rounded-full shadow-2xl flex items-center gap-4 border font-bold text-base', 
                    notification.type === 'success' ? 'bg-emerald-500/90 text-white border-emerald-400' : 'bg-rose-500/90 text-white border-rose-400']">
        <svg v-if="notification.type === 'success'" xmlns="http://www.w3.org/2000/svg" class="w-6 h-6" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>
        <svg v-else xmlns="http://www.w3.org/2000/svg" class="w-6 h-6" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>
        {{ notification.message }}
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted, watch, reactive } from 'vue'
import { store } from '../store'

const isTraining = reactive<Record<string, boolean>>({
  traffic: false,
  optical: false
})



const trainConfigs = ref<Record<string, any>>({
  traffic: { epochs: 100, learning_rate: 0.001, batch_size: 32, threshold_percentile: 99.99, patience: 10 },
  optical: { epochs: 100, learning_rate: 0.001, batch_size: 32, threshold_percentile: 99.99, patience: 10 }
})

const getPastDate = (days: number) => {
  const d = new Date()
  d.setDate(d.getDate() - days)
  return d.toISOString().split('T')[0]
}

const useRecentWindow = ref<Record<string, boolean>>({ traffic: true, optical: true })   // true: 날짜 미지정 -> 서버가 최근 구간 포트 분할
const excludeSuspect = ref<Record<string, boolean>>({ traffic: true, optical: true })
const alertPolicy = ref<Record<string, string>>({ traffic: '', optical: '' })   // '' = 활성 모델 정책 승계
const dateConfigs = ref<Record<string, any>>({
  traffic: { train_start: getPastDate(37), train_end: getPastDate(7), test_start: getPastDate(7), test_end: getPastDate(0) },
  optical: { train_start: getPastDate(37), train_end: getPastDate(7), test_start: getPastDate(7), test_end: getPastDate(0) }
})



const notification = reactive({
  show: false,
  message: '',
  type: 'success'
})

let notifTimeout: ReturnType<typeof setTimeout> | null = null

const showNotification = (msg: string, type: 'success' | 'error' = 'success') => {
  notification.message = msg
  notification.type = type
  notification.show = true
  if (notifTimeout) clearTimeout(notifTimeout)
  notifTimeout = setTimeout(() => {
    notification.show = false
  }, 3000)
}

const handleCheckDrift = async () => {
  try {
    const data = await store.checkDrift()
    showNotification('Drift check completed.', 'success')
    if (data.auto_retrain_triggered) {
      showNotification('Auto-Retraining triggered!', 'success')
      setTimeout(async () => {
        await store.fetchModelStatus()
        startPolling()
      }, 1000)
    }
  } catch (err: any) {
    const errMsg = err.response?.data?.detail || err.message || 'Failed to check drift'
    showNotification(errMsg, 'error')
  }
}

watch(() => store.modelStatus, (newVal) => {
  if (newVal.traffic) {
    trainConfigs.value.traffic = { ...newVal.traffic.training_config }
    isTraining.traffic = newVal.traffic.training?.is_training || false
  }
  if (newVal.optical) {
    trainConfigs.value.optical = { ...newVal.optical.training_config }
    isTraining.optical = newVal.optical.training?.is_training || false
  }
}, { deep: true, immediate: true })
let statusTimer: ReturnType<typeof setInterval> | null = null

const stopPolling = () => {
  if (statusTimer) {
    clearInterval(statusTimer)
    statusTimer = null
    console.log('[Poll] Stopped polling (No active training)')
  }
}

const startPolling = () => {
  if (statusTimer) return // Already polling
  console.log('[Poll] Started polling for training progress')
  statusTimer = setInterval(async () => {
    await store.fetchModelStatus()
    
    // 모든 모델의 학습 상태 확인
    const isAnyTraining = Object.values(store.modelStatus).some((m: any) => m.training?.is_training)
    if (!isAnyTraining) {
      stopPolling()
      await store.fetchModelVersions()      // 학습이 끝나면 새 후보가 목록에 나타남
    }
  }, 10000)
}

const handleTrain = async (ft: string) => {
  const dates = useRecentWindow.value[ft] ? {} : dateConfigs.value[ft]
  if (!confirm(`설정한 파라미터로 후보 모델을 학습하시겠습니까?\n(결과는 Candidate 로 저장되며 승격 전에는 실시간 엔진에 반영되지 않습니다)`)) return
  
  isTraining[ft] = true 
  try {
    await store.trainModel(ft, trainConfigs.value[ft], dates, excludeSuspect.value[ft], alertPolicy.value[ft])
    showNotification(`${ft.toUpperCase()} training task started.`, 'success')
    await store.fetchModelStatus()
    startPolling() // 학습 시작 시 폴링 시작
  } catch (err: any) {
    const errMsg = err.response?.data?.detail || err.message || `Failed to start ${ft} training.`
    showNotification(errMsg, 'error')
    isTraining[ft] = false
  }
}

// --- Model Versions ---
const versionsOf = (ft: string): any[] => {
  const list = store.modelVersions?.[ft]?.versions || []
  return [...list].reverse()          // 최신 버전이 위로
}

const canRollback = (ft: string) => {
  const list = store.modelVersions?.[ft]?.versions || []
  return list.some((x: any) => x.status === 'retired')
}

// --- Drift / Gate 표시 ---
const driftBadge = (ft: string) => {
  const st = store.driftStatus?.[ft]?.status
  if (st === 'drifted') return { text: 'Widespread Drift', cls: 'bg-rose-500/20 text-rose-400' }
  if (st === 'localized') return { text: 'Localized (fault?)', cls: 'bg-amber-500/20 text-amber-400' }
  if (st === 'warming_up') return { text: 'Warming up', cls: 'bg-sky-500/20 text-sky-400' }
  if (st && st !== 'normal') return { text: st.replace('_', ' '), cls: 'bg-slate-500/20 text-slate-400' }
  return { text: 'Normal', cls: 'bg-emerald-500/20 text-emerald-400' }
}

const gateClass = (status: string | null | undefined) => {
  if (status === 'PASS') return 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30'
  if (status === 'WARN') return 'bg-amber-500/20 text-amber-400 border-amber-500/30'
  if (status === 'FAIL' || status === 'ERROR') return 'bg-rose-500/20 text-rose-400 border-rose-500/30'
  return 'bg-slate-500/20 text-slate-400 border-slate-500/30'          // SKIP / 미실행
}

const openGates = ref<Record<string, boolean>>({})
const gateKey = (ft: string, version: string) => `${ft}:${version}`
const toggleGate = (ft: string, version: string) => { openGates.value[gateKey(ft, version)] = !openGates.value[gateKey(ft, version)] }
const isGateOpen = (ft: string, version: string) => !!openGates.value[gateKey(ft, version)]

const handleRerunGate = async (ft: string, version: string) => {
  try {
    const r = await store.rerunGate(ft, version)
    openGates.value[gateKey(ft, version)] = true
    showNotification(`${ft.toUpperCase()} ${version} 게이트: ${r.gate.status}`, r.gate.status === 'PASS' ? 'success' : 'error')
  } catch (err: any) {
    const detail = err.response?.data?.detail
    showNotification((typeof detail === 'string' ? detail : null) || err.message || '게이트 실행 실패', 'error')
  }
}

const fmtNum = (n: number | null | undefined) => (n === null || n === undefined) ? '-' : Number(n).toPrecision(4)

const handlePromote = async (ft: string, version: string) => {
  if (!confirm(`${ft.toUpperCase()} 모델 ${version} 을(를) 활성화합니다.\n실시간 추론 엔진에 즉시 반영됩니다. 계속하시겠습니까?`)) return
  try {
    await store.promoteModel(ft, version, false)
    showNotification(`${ft.toUpperCase()} ${version} 활성화 완료`, 'success')
    await store.fetchModelStatus()
  } catch (err: any) {
    const detail = err.response?.data?.detail
    if (err.response?.status === 409 && detail?.warnings) {
      // 서버가 현재 모델과 크게 다른 후보라고 경고: 내용을 보여주고 한 번 더 확인
      const list = detail.warnings.map((w: string) => `- ${w}`).join('\n')
      if (!confirm(`경고: 승격 검증에서 문제가 발견되었습니다.\n\n${list}\n\n그래도 활성화하시겠습니까?`)) return
      try {
        await store.promoteModel(ft, version, true)
        showNotification(`${ft.toUpperCase()} ${version} 활성화 완료 (경고 무시)`, 'success')
        await store.fetchModelStatus()
      } catch (e2: any) {
        showNotification(e2.response?.data?.detail || e2.message || '승격 실패', 'error')
      }
      return
    }
    showNotification((typeof detail === 'string' ? detail : null) || err.message || '승격 실패', 'error')
  }
}

const handleRollback = async (ft: string) => {
  if (!confirm(`${ft.toUpperCase()} 모델을 직전 활성 버전으로 되돌립니다. 계속하시겠습니까?`)) return
  try {
    const r = await store.rollbackModel(ft)
    showNotification(`${ft.toUpperCase()} ${r.previous} → ${r.active} 롤백 완료`, 'success')
    await store.fetchModelStatus()
  } catch (err: any) {
    showNotification(err.response?.data?.detail || err.message || '롤백 실패', 'error')
  }
}

const handleStop = async (ft: string) => {
  if (!confirm(`훈련을 중지하시겠습니까?`)) return
  try {
    await store.stopTraining(ft)
    showNotification(`${ft.toUpperCase()} training stop requested.`, 'success')
    await store.fetchModelStatus()
    // handleStop 후에도 폴링은 계속됨 (서버에서 완전히 멈출 때까지 기다림)
  } catch (err: any) {
    const errMsg = err.response?.data?.detail || err.message || `Failed to stop ${ft} training.`
    showNotification(errMsg, 'error')
  }
}

onMounted(async () => {
  await store.fetchModelStatus()
  await store.fetchModelVersions()
  await store.fetchDriftStatus()
  
  // 진입 시 이미 학습 중인 모델이 있다면 폴링 시작
  const isAnyTraining = Object.values(store.modelStatus).some((m: any) => m.training?.is_training)
  if (isAnyTraining) {
    startPolling()
  }
})

onUnmounted(() => {
  if (notifTimeout) clearTimeout(notifTimeout)
  stopPolling()
})
</script>

<style scoped>
input[type=range]::-webkit-slider-thumb {
  -webkit-appearance: none;
  height: 20px;
  width: 20px;
  border-radius: 50%;
  background: white;
  cursor: pointer;
  margin-top: -6px;
  box-shadow: 0 0 10px rgba(0,0,0,0.5);
}
</style>
