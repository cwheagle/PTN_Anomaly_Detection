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
                </div>

                <div class="space-y-4 pt-4">
                  <div class="flex items-center gap-3">
                    <div class="w-1.5 h-4 bg-amber-500/50 rounded-full"></div>
                    <span class="text-xs font-bold text-slate-400 uppercase tracking-wider">Dataset Range</span>
                  </div>
                  
                  <div class="grid grid-cols-2 gap-6 bg-slate-900/50 p-4 rounded-xl border border-slate-700/50">
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

                <button @click="handleTrain(ft as string)" 
                        :disabled="isTraining[ft as string]"
                        class="w-full py-4 bg-amber-600/10 hover:bg-amber-600/20 border border-amber-500/30 text-amber-500 disabled:opacity-50 disabled:cursor-not-allowed rounded-xl text-sm font-bold transition-all flex items-center justify-center gap-3">
                  <svg v-if="isTraining[ft as string]" class="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                  {{ isTraining[ft as string] ? 'Training...' : 'Train' }}
                </button>
              </div>
            </div>

            <!-- 2. Inference Configuration (Bottom) -->
            <div class="space-y-8">
              <div class="flex items-center gap-3">
                <div class="w-2 h-5 bg-emerald-500 rounded-full"></div>
                <h4 class="text-sm font-bold text-slate-200 uppercase tracking-wider">Inference Configuration</h4>
              </div>
              
              <div class="space-y-8 bg-slate-800/30 p-6 rounded-xl border border-slate-700/50">
                <div class="space-y-3">
                  <div class="flex justify-between items-center">
                    <label class="text-xs text-slate-400 uppercase font-bold tracking-wider">Anomaly Threshold (MSE)</label>
                    <span class="text-sm font-mono text-emerald-400 font-bold bg-emerald-500/10 px-2 py-1 rounded">{{ infConfigs[ft]?.threshold?.toFixed(4) }}</span>
                  </div>
                  <input type="range" v-model.number="infConfigs[ft].threshold" min="0.0001" max="0.5" step="0.0001"
                         class="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-emerald-500" />
                </div>

                <div class="space-y-3">
                  <div class="flex justify-between items-center">
                    <label class="text-xs text-slate-400 uppercase font-bold tracking-wider">Trend Sensitivity (Slope)</label>
                    <span class="text-sm font-mono text-emerald-400 font-bold bg-emerald-500/10 px-2 py-1 rounded">{{ infConfigs[ft]?.slope_threshold?.toFixed(1) }}</span>
                  </div>
                  <input type="range" v-model.number="infConfigs[ft].slope_threshold" min="0.1" max="10.0" step="0.1"
                         class="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-emerald-500" />
                </div>

                <button @click="handleSaveInference(ft as string)"
                        :disabled="!infDirty[ft as string]"
                        :class="['w-full py-3.5 rounded-xl text-sm font-bold transition-all border mt-4', 
                                 infDirty[ft as string] ? 'bg-emerald-600/20 hover:bg-emerald-600/30 text-emerald-400 border-emerald-500/30 shadow-lg shadow-emerald-500/10' : 'bg-slate-800 text-slate-600 border-slate-700 cursor-not-allowed']">
                  Save
                </button>
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
import { ref, onMounted, watch, reactive } from 'vue'
import { store } from '../store'

const isTraining = reactive<Record<string, boolean>>({
  traffic: false,
  optical: false
})

const infConfigs = ref<Record<string, any>>({
  traffic: { threshold: 0.1, slope_threshold: 1.0 },
  optical: { threshold: 0.1, slope_threshold: 1.0 }
})

const trainConfigs = ref<Record<string, any>>({
  traffic: { epochs: 100, learning_rate: 0.001, batch_size: 32, threshold_percentile: 99.9 },
  optical: { epochs: 100, learning_rate: 0.001, batch_size: 32, threshold_percentile: 99.9 }
})

const getPastDate = (days: number) => {
  const d = new Date()
  d.setDate(d.getDate() - days)
  return d.toISOString().split('T')[0]
}

const dateConfigs = ref<Record<string, any>>({
  traffic: { train_start: getPastDate(37), train_end: getPastDate(7), test_start: getPastDate(7), test_end: getPastDate(0) },
  optical: { train_start: getPastDate(37), train_end: getPastDate(7), test_start: getPastDate(7), test_end: getPastDate(0) }
})

const infDirty = reactive<Record<string, boolean>>({
  traffic: false,
  optical: false
})

const notification = reactive({
  show: false,
  message: '',
  type: 'success'
})

const showNotice = (msg: string, type = 'success') => {
  notification.message = msg
  notification.type = type
  notification.show = true
  setTimeout(() => notification.show = false, 3000)
}

watch(() => store.modelStatus, (newVal) => {
  if (newVal.traffic) {
    infConfigs.value.traffic = { ...newVal.traffic.inference_config }
    trainConfigs.value.traffic = { ...newVal.traffic.training_config }
    infDirty.traffic = false
  }
  if (newVal.optical) {
    infConfigs.value.optical = { ...newVal.optical.inference_config }
    trainConfigs.value.optical = { ...newVal.optical.training_config }
    infDirty.optical = false
  }
}, { deep: true, immediate: true })

watch(() => infConfigs.value.traffic, () => { infDirty.traffic = true }, { deep: true })
watch(() => infConfigs.value.optical, () => { infDirty.optical = true }, { deep: true })

const handleSaveInference = async (ft: string) => {
  try {
    await store.updateInferenceConfig(ft, infConfigs.value[ft])
    showNotice(`${ft.toUpperCase()} inference configuration applied.`)
    infDirty[ft] = false
  } catch (err) {
    showNotice(`Failed to update ${ft} configuration.`, 'error')
  }
}

const handleTrain = async (ft: string) => {
  const dates = dateConfigs.value[ft]
  if (!confirm(`Trigger ${ft} model training?`)) return
  isTraining[ft] = true
  try {
    await store.trainModel(ft, trainConfigs.value[ft], dates)
    showNotice(`${ft.toUpperCase()} training task started.`)
  } catch (err) {
    showNotice(`Failed to start ${ft} training.`, 'error')
    isTraining[ft] = false
  }
}

onMounted(async () => {
  await store.fetchModelStatus()
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
