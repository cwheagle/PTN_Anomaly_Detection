<template>
  <div class="space-y-6">
    <div class="flex justify-between items-center bg-slate-800 p-6 rounded-xl border border-slate-700 shadow-xl">
      <div>
        <h2 class="text-xl font-bold text-blue-400 flex items-center gap-2">
          <svg xmlns="http://www.w3.org/2000/svg" class="w-6 h-6" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
          RCA Domain Rules
        </h2>
        <p class="text-sm text-slate-400 mt-1">Manage structured domain rules (contributions and raw conditions) to improve Root Cause Analysis precision.</p>
      </div>
      <button @click="openAddModal" class="px-5 py-2.5 bg-blue-600 hover:bg-blue-500 text-white text-sm font-bold rounded-lg shadow-lg shadow-blue-500/20 transition-all flex items-center gap-2">
        <svg xmlns="http://www.w3.org/2000/svg" class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="5" x2="12" y2="19"></line><line x1="5" y1="12" x2="19" y2="12"></line></svg>
        Add Rule
      </button>
    </div>

    <!-- Rules Table -->
    <div class="bg-slate-800 rounded-xl border border-slate-700 shadow-xl overflow-hidden flex flex-col min-h-[500px]">
      <div class="flex justify-between items-center p-4 border-b border-slate-700 bg-slate-800/50">
        <h3 class="font-bold text-slate-200">Active Rule Set</h3>
        <button @click="store.fetchRcaRules()" class="p-2 text-slate-400 hover:text-blue-400 hover:bg-blue-500/10 rounded-lg transition-colors">
          <svg xmlns="http://www.w3.org/2000/svg" class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"></polyline><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path></svg>
        </button>
      </div>
      <div class="overflow-x-auto flex-1">
        <table class="w-full text-left text-sm">
          <thead class="bg-slate-700/30 text-slate-400 uppercase text-[11px] tracking-wider">
            <tr>
              <th class="px-6 py-4">Rule ID</th>
              <th class="px-6 py-4">Track</th>
              <th class="px-6 py-4">Priority</th>
              <th class="px-6 py-4 max-w-sm">Conditions</th>
              <th class="px-6 py-4">Diagnosis</th>
              <th class="px-6 py-4">Action</th>
              <th class="px-6 py-4 text-right">Delete</th>
            </tr>
          </thead>
          <tbody class="divide-y divide-slate-700">
            <tr v-for="rule in sortedRules" :key="rule.id" class="hover:bg-slate-700/20 transition-colors">
              <td class="px-6 py-4 font-mono font-bold text-slate-300">{{ rule.id }}</td>
              <td class="px-6 py-4">
                <span :class="['px-2 py-1 rounded text-[10px] font-bold uppercase', 
                              rule.track === 'traffic' ? 'bg-blue-500/20 text-blue-400' : 
                              rule.track === 'optical' ? 'bg-purple-500/20 text-purple-400' : 'bg-emerald-500/20 text-emerald-400']">
                  {{ rule.track }}
                </span>
              </td>
              <td class="px-6 py-4 font-mono text-slate-400">{{ rule.priority }}</td>
              <td class="px-6 py-4">
                <div class="flex flex-wrap gap-1.5">
                  <span v-for="(v, k) in rule.contributions" :key="'c_'+k" 
                        class="px-1.5 py-0.5 bg-indigo-500/10 border border-indigo-500/30 text-indigo-300 text-[10px] rounded">
                    [기여도] {{ k }} ≥ {{ v }}%
                  </span>
                  <span v-for="(v, k) in rule.raw_conditions" :key="'r_'+k" 
                        class="px-1.5 py-0.5 bg-amber-500/10 border border-amber-500/30 text-amber-300 text-[10px] rounded">
                    [원시] {{ String(k).replace('min_', '').replace('max_', '') }} 
                    {{ String(k).startsWith('min_') ? '≥' : '≤' }} {{ v }}
                  </span>
                  <span v-if="!rule.contributions && !rule.raw_conditions" class="text-xs text-slate-500 italic">No conditions</span>
                </div>
              </td>
              <td class="px-6 py-4 font-bold text-slate-300">{{ rule.diagnosis }}</td>
              <td class="px-6 py-4 text-emerald-400 text-xs truncate max-w-[150px]" :title="rule.action">{{ rule.action }}</td>
              <td class="px-6 py-4 text-right">
                <button @click="deleteRule(rule.id)" class="text-slate-500 hover:text-rose-400 transition-colors p-1" title="Delete Rule">
                  <svg xmlns="http://www.w3.org/2000/svg" class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path><line x1="10" y1="11" x2="10" y2="17"></line><line x1="14" y1="11" x2="14" y2="17"></line></svg>
                </button>
              </td>
            </tr>
            <tr v-if="sortedRules.length === 0">
              <td colspan="7" class="px-6 py-12 text-center text-slate-500 italic">No rules defined. System will use default fallback.</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- Add Rule Modal (Dynamic Form) -->
    <div v-if="showAddModal" class="fixed inset-0 bg-slate-950/60 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div class="bg-slate-800/95 border border-slate-700/50 rounded-2xl w-full max-w-4xl shadow-2xl animate-in zoom-in duration-200 flex flex-col max-h-[90vh]">
        <div class="p-6 border-b border-slate-700/50 flex justify-between items-center shrink-0">
          <h3 class="text-xl font-bold text-blue-400 flex items-center gap-2">
            <svg xmlns="http://www.w3.org/2000/svg" class="w-5 h-5 text-blue-500" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 2 7 12 12 22 7 12 2"></polygon><polyline points="2 17 12 22 22 17"></polyline><polyline points="2 12 12 17 22 12"></polyline></svg>
            Add Structured Domain Rule
          </h3>
          <button @click="closeModal" class="text-slate-400 hover:text-slate-200">
            <svg xmlns="http://www.w3.org/2000/svg" class="w-6 h-6" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
          </button>
        </div>
        
        <form @submit.prevent="submitRule" class="p-6 overflow-y-auto space-y-6">
          <div class="grid grid-cols-3 gap-5">
            <div class="space-y-1.5">
              <label class="text-xs font-bold text-slate-400 uppercase tracking-wide">Rule ID</label>
              <input v-model="form.id" type="text" required placeholder="e.g. IN-001" class="w-full bg-slate-900/50 border border-slate-700 rounded-lg px-4 py-2 text-sm text-slate-200 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all">
            </div>
            <div class="space-y-1.5">
              <label class="text-xs font-bold text-slate-400 uppercase tracking-wide">Track</label>
              <select v-model="form.track" @change="onTrackChange" required class="w-full bg-slate-900/50 border border-slate-700 rounded-lg px-4 py-2 text-sm text-slate-200 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all">
                <option value="traffic">Traffic</option>
                <option value="optical">Optical</option>
                <option value="integrated">Integrated</option>
              </select>
            </div>
            <div class="space-y-1.5">
              <label class="text-xs font-bold text-slate-400 uppercase tracking-wide">Priority (Higher runs first)</label>
              <input v-model.number="form.priority" type="number" required class="w-full bg-slate-900/50 border border-slate-700 rounded-lg px-4 py-2 text-sm text-slate-200 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all">
            </div>
          </div>
          
          <!-- Dynamic Feature Conditions -->
          <div class="space-y-3">
            <label class="text-xs font-bold text-slate-400 uppercase tracking-wide border-b border-slate-700 pb-2 block">
              Feature Conditions
            </label>
            <div class="grid grid-cols-1 gap-3">
              <div v-for="(cfg, feat) in featureConfig" :key="feat" 
                   class="p-4 rounded-lg border transition-all"
                   :class="cfg.enabled ? 'bg-blue-900/10 border-blue-500/30' : 'bg-slate-800/50 border-slate-700'">
                
                <div class="flex items-center gap-3 mb-3">
                  <input type="checkbox" :id="'chk_'+feat" v-model="cfg.enabled" class="w-4 h-4 accent-blue-500 bg-slate-900 border-slate-700 rounded">
                  <label :for="'chk_'+feat" class="font-bold cursor-pointer" :class="cfg.enabled ? 'text-blue-300' : 'text-slate-500'">
                    {{ feat }}
                  </label>
                </div>

                <div v-if="cfg.enabled" class="flex flex-col gap-4 animate-in fade-in slide-in-from-top-2">
                  <!-- Row 1: Contrib & Raw -->
                  <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div class="space-y-1">
                      <span class="text-[10px] text-indigo-400 font-bold uppercase tracking-wider">Contrib (%) Min</span>
                      <input v-model="cfg.min_contrib" type="number" step="0.1" placeholder="e.g. 50" class="w-full bg-slate-900 border border-indigo-500/30 rounded px-3 py-1.5 text-sm text-slate-200 focus:outline-none focus:border-indigo-500 transition-all">
                    </div>
                    <div class="space-y-1">
                      <span class="text-[10px] text-amber-400 font-bold uppercase tracking-wider">Raw Min</span>
                      <input v-model="cfg.min_raw" type="number" step="any" placeholder="e.g. 100" class="w-full bg-slate-900 border border-amber-500/30 rounded px-3 py-1.5 text-sm text-slate-200 focus:outline-none focus:border-amber-500 transition-all">
                    </div>
                    <div class="space-y-1">
                      <span class="text-[10px] text-amber-400 font-bold uppercase tracking-wider">Raw Max</span>
                      <input v-model="cfg.max_raw" type="number" step="any" placeholder="e.g. 1000" class="w-full bg-slate-900 border border-amber-500/30 rounded px-3 py-1.5 text-sm text-slate-200 focus:outline-none focus:border-amber-500 transition-all">
                    </div>
                  </div>
                  <!-- Row 2: Ratio & Slope (Phase 9) -->
                  <div class="grid grid-cols-2 md:grid-cols-4 gap-4 pt-2 border-t border-slate-700/50">
                    <div class="space-y-1">
                      <span class="text-[10px] text-emerald-400 font-bold uppercase tracking-wider">Ratio Min</span>
                      <input v-model="cfg.min_ratio" :disabled="feat.includes('power')" type="number" step="any" placeholder="e.g. 0.5" class="w-full bg-slate-900 border border-emerald-500/30 rounded px-3 py-1.5 text-sm text-slate-200 focus:outline-none focus:border-emerald-500 disabled:opacity-30 transition-all" title="Ratio to moving avg (Not applicable for dBm power)">
                    </div>
                    <div class="space-y-1">
                      <span class="text-[10px] text-emerald-400 font-bold uppercase tracking-wider">Ratio Max</span>
                      <input v-model="cfg.max_ratio" :disabled="feat.includes('power')" type="number" step="any" placeholder="e.g. 1.5" class="w-full bg-slate-900 border border-emerald-500/30 rounded px-3 py-1.5 text-sm text-slate-200 focus:outline-none focus:border-emerald-500 disabled:opacity-30 transition-all">
                    </div>
                    <div class="space-y-1">
                      <span class="text-[10px] text-purple-400 font-bold uppercase tracking-wider">Trend Slope Min</span>
                      <input v-model="cfg.min_slope" type="number" step="any" placeholder="e.g. -0.5" class="w-full bg-slate-900 border border-purple-500/30 rounded px-3 py-1.5 text-sm text-slate-200 focus:outline-none focus:border-purple-500 transition-all">
                    </div>
                    <div class="space-y-1">
                      <span class="text-[10px] text-purple-400 font-bold uppercase tracking-wider">Trend Slope Max</span>
                      <input v-model="cfg.max_slope" type="number" step="any" placeholder="e.g. 0.5" class="w-full bg-slate-900 border border-purple-500/30 rounded px-3 py-1.5 text-sm text-slate-200 focus:outline-none focus:border-purple-500 transition-all">
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <!-- Track Severity Conditions -->
          <div class="space-y-3">
            <label class="text-xs font-bold text-slate-400 uppercase tracking-wide border-b border-slate-700 pb-2 block">
              Track Severity Conditions (Optional)
            </label>
            <div class="grid grid-cols-2 gap-5 p-4 rounded-lg bg-slate-800/50 border border-slate-700">
              <div v-if="form.track === 'traffic' || form.track === 'integrated'" class="space-y-3">
                <span class="text-sm font-bold text-blue-300">Traffic Severity (0-100)</span>
                <div class="grid grid-cols-2 gap-4">
                  <div class="space-y-1">
                    <span class="text-[10px] text-amber-400 font-bold uppercase tracking-wider">Min</span>
                    <input v-model="form.min_traffic_severity" type="number" step="any" placeholder="e.g. 70" class="w-full bg-slate-900 border border-amber-500/30 rounded px-3 py-1.5 text-sm text-slate-200 focus:outline-none focus:border-amber-500 transition-all">
                  </div>
                  <div class="space-y-1">
                    <span class="text-[10px] text-amber-400 font-bold uppercase tracking-wider">Max</span>
                    <input v-model="form.max_traffic_severity" type="number" step="any" placeholder="e.g. 100" class="w-full bg-slate-900 border border-amber-500/30 rounded px-3 py-1.5 text-sm text-slate-200 focus:outline-none focus:border-amber-500 transition-all">
                  </div>
                </div>
              </div>
              <div v-if="form.track === 'optical' || form.track === 'integrated'" class="space-y-3">
                <span class="text-sm font-bold text-purple-300">Optical Severity (0-100)</span>
                <div class="grid grid-cols-2 gap-4">
                  <div class="space-y-1">
                    <span class="text-[10px] text-amber-400 font-bold uppercase tracking-wider">Min</span>
                    <input v-model="form.min_optical_severity" type="number" step="any" placeholder="e.g. 70" class="w-full bg-slate-900 border border-amber-500/30 rounded px-3 py-1.5 text-sm text-slate-200 focus:outline-none focus:border-amber-500 transition-all">
                  </div>
                  <div class="space-y-1">
                    <span class="text-[10px] text-amber-400 font-bold uppercase tracking-wider">Max</span>
                    <input v-model="form.max_optical_severity" type="number" step="any" placeholder="e.g. 100" class="w-full bg-slate-900 border border-amber-500/30 rounded px-3 py-1.5 text-sm text-slate-200 focus:outline-none focus:border-amber-500 transition-all">
                  </div>
                </div>
              </div>
            </div>
          </div>
          
          <div class="grid grid-cols-2 gap-5">
            <div class="space-y-1.5">
              <label class="text-xs font-bold text-slate-400 uppercase tracking-wide">Diagnosis Result</label>
              <textarea v-model="form.diagnosis" required rows="2" placeholder="e.g. CRC Error Detected" class="w-full bg-slate-900/50 border border-slate-700 rounded-lg px-4 py-2 text-sm text-slate-200 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all resize-none"></textarea>
            </div>
            <div class="space-y-1.5">
              <label class="text-xs font-bold text-slate-400 uppercase tracking-wide text-emerald-400">Recommended Action</label>
              <textarea v-model="form.action" required rows="2" placeholder="e.g. Inspect optical fiber connectors" class="w-full bg-emerald-900/10 border border-emerald-500/30 rounded-lg px-4 py-2 text-sm text-emerald-200 focus:outline-none focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500 transition-all resize-none"></textarea>
            </div>
          </div>
          
          <div v-if="errorMsg" class="p-3 bg-rose-500/10 border border-rose-500/30 rounded-lg text-xs text-rose-400 flex items-center gap-2">
            <svg xmlns="http://www.w3.org/2000/svg" class="w-4 h-4 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>
            {{ errorMsg }}
          </div>

          <div class="flex justify-end gap-3 pt-4 border-t border-slate-700/50 shrink-0">
            <button type="button" @click="closeModal" class="px-5 py-2 text-sm font-bold text-slate-300 hover:bg-slate-700 rounded-lg transition-colors">Cancel</button>
            <button type="submit" :disabled="isSubmitting" class="px-5 py-2 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white text-sm font-bold rounded-lg shadow-lg shadow-blue-500/20 transition-all flex items-center gap-2">
              <svg v-if="isSubmitting" class="animate-spin -ml-1 mr-1 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
              Save Structured Rule
            </button>
          </div>
        </form>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { store } from '../store'

const showAddModal = ref(false)
const isSubmitting = ref(false)
const errorMsg = ref('')

const TRACK_FEATURES: Record<string, string[]> = {
  traffic: ['tx_packet', 'rx_packet', 'error_packet'],
  optical: ['tx_avg_power', 'rx_avg_power'],
  integrated: ['tx_packet', 'rx_packet', 'error_packet', 'tx_avg_power', 'rx_avg_power']
}

const defaultForm = {
  id: '',
  track: 'traffic',
  priority: 10,
  diagnosis: '',
  action: '',
  min_traffic_severity: null as number | null,
  max_traffic_severity: null as number | null,
  min_optical_severity: null as number | null,
  max_optical_severity: null as number | null
}

const form = ref({ ...defaultForm })
const featureConfig = ref<Record<string, any>>({})

const onTrackChange = () => {
  featureConfig.value = {}
  TRACK_FEATURES[form.value.track].forEach(feat => {
    featureConfig.value[feat] = {
      enabled: false,
      min_contrib: null,
      min_raw: null,
      max_raw: null,
      min_ratio: null,
      max_ratio: null,
      min_slope: null,
      max_slope: null
    }
  })
}

const openAddModal = () => {
  form.value = { ...defaultForm }
  onTrackChange()
  showAddModal.value = true
}

const closeModal = () => {
  showAddModal.value = false
  errorMsg.value = ''
}

onMounted(() => {
  store.fetchRcaRules()
})

const sortedRules = computed(() => {
  return [...store.rcaRules].sort((a, b) => b.priority - a.priority)
})

const submitRule = async () => {
  try {
    isSubmitting.value = true
    errorMsg.value = ''

    // Assemble payload
    const payload: any = {
      id: form.value.id,
      track: form.value.track,
      priority: form.value.priority,
      diagnosis: form.value.diagnosis,
      action: form.value.action,
      contributions: {},
      raw_conditions: {}
    }

    let hasCondition = false

    for (const [feat, cfg] of Object.entries(featureConfig.value)) {
      if (cfg.enabled) {
        if (cfg.min_contrib !== null && cfg.min_contrib !== '') {
          payload.contributions[feat] = Number(cfg.min_contrib)
          hasCondition = true
        }
        if (cfg.min_raw !== null && cfg.min_raw !== '') {
          payload.raw_conditions[`min_${feat}`] = Number(cfg.min_raw)
          hasCondition = true
        }
        if (cfg.max_raw !== null && cfg.max_raw !== '') {
          payload.raw_conditions[`max_${feat}`] = Number(cfg.max_raw)
          hasCondition = true
        }
        if (cfg.min_ratio !== null && cfg.min_ratio !== '') {
          payload.raw_conditions[`min_${feat}_ratio`] = Number(cfg.min_ratio)
          hasCondition = true
        }
        if (cfg.max_ratio !== null && cfg.max_ratio !== '') {
          payload.raw_conditions[`max_${feat}_ratio`] = Number(cfg.max_ratio)
          hasCondition = true
        }
        if (cfg.min_slope !== null && cfg.min_slope !== '') {
          payload.raw_conditions[`min_${feat}_trend_slope`] = Number(cfg.min_slope)
          hasCondition = true
        }
        if (cfg.max_slope !== null && cfg.max_slope !== '') {
          payload.raw_conditions[`max_${feat}_trend_slope`] = Number(cfg.max_slope)
          hasCondition = true
        }
      }
    }

    // Append Track Severity Conditions
    if (form.value.min_traffic_severity !== null && (form.value.min_traffic_severity as any) !== '') {
      payload.raw_conditions['min_traffic_severity'] = Number(form.value.min_traffic_severity)
      hasCondition = true
    }
    if (form.value.max_traffic_severity !== null && (form.value.max_traffic_severity as any) !== '') {
      payload.raw_conditions['max_traffic_severity'] = Number(form.value.max_traffic_severity)
      hasCondition = true
    }
    if (form.value.min_optical_severity !== null && (form.value.min_optical_severity as any) !== '') {
      payload.raw_conditions['min_optical_severity'] = Number(form.value.min_optical_severity)
      hasCondition = true
    }
    if (form.value.max_optical_severity !== null && (form.value.max_optical_severity as any) !== '') {
      payload.raw_conditions['max_optical_severity'] = Number(form.value.max_optical_severity)
      hasCondition = true
    }

    if (!hasCondition) {
      errorMsg.value = "Please enable and configure at least one feature condition."
      isSubmitting.value = false
      return
    }

    // Clean up empty objects
    if (Object.keys(payload.contributions).length === 0) delete payload.contributions
    if (Object.keys(payload.raw_conditions).length === 0) delete payload.raw_conditions

    await store.addRcaRule(payload)
    closeModal()
  } catch (err: any) {
    errorMsg.value = err.response?.data?.detail || err.message || 'Failed to save rule'
  } finally {
    isSubmitting.value = false
  }
}

const deleteRule = async (id: string) => {
  if (confirm(`Are you sure you want to delete rule '${id}'?`)) {
    try {
      await store.deleteRcaRule(id)
    } catch (err: any) {
      alert('Failed to delete rule: ' + (err.response?.data?.detail || err.message))
    }
  }
}
</script>
