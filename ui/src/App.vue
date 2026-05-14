<template>
  <div class="min-h-screen bg-slate-900 text-slate-100 p-6 flex flex-col">
    <header class="mb-4 flex justify-between items-center border-b border-slate-700 pb-4">
      <div>
        <h1 class="text-2xl font-bold text-blue-400">PTN Anomaly Detection Dashboard</h1>
        <p class="text-slate-400 text-sm">Real-time Network Health Monitoring</p>
      </div>
      <div class="flex items-center gap-4">
        <!-- Notification Bell -->
        <div class="relative">
          <button @click="showNotificationCenter = !showNotificationCenter" 
                  class="p-2 bg-slate-800 hover:bg-slate-700 rounded-full border border-slate-700 transition-all relative">
            <svg xmlns="http://www.w3.org/2000/svg" class="w-5 h-5 text-slate-300" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"></path><path d="M13.73 21a2 2 0 0 1-3.46 0"></path></svg>
            <span v-if="store.alarms.length > 0" class="absolute top-0 right-0 w-2.5 h-2.5 bg-rose-500 border-2 border-slate-900 rounded-full"></span>
          </button>
          
          <!-- Notification Dropdown -->
          <div v-if="showNotificationCenter" 
               class="absolute right-0 mt-3 w-80 bg-slate-800 border border-slate-700 rounded-xl shadow-2xl z-[100] overflow-hidden animate-in fade-in zoom-in duration-200">
            <div class="p-4 bg-slate-700/50 border-b border-slate-700 flex justify-between items-center">
              <h3 class="font-bold text-sm text-slate-200">Critical Alarms History</h3>
              <button @click="store.alarms = []" class="text-[10px] text-slate-500 hover:text-rose-400 font-bold uppercase">Clear</button>
            </div>
            <div class="max-h-96 overflow-y-auto p-2 space-y-2">
              <div v-for="(alarm, idx) in store.alarms" :key="idx" class="p-3 bg-slate-900/50 border border-slate-700 rounded-lg hover:border-rose-500/30 transition-colors">
                <div class="flex justify-between items-start mb-1">
                  <span class="text-[10px] font-mono text-slate-500">{{ alarm.event_time }}</span>
                  <span class="text-[9px] bg-rose-500/20 text-rose-400 px-1 rounded font-bold uppercase">Critical</span>
                </div>
                <p class="text-xs font-bold text-slate-300">{{ alarm.ip_addr }} (S{{ alarm.slot_id }}/P{{ alarm.port_id }})</p>
                <p class="text-[11px] text-slate-500 mt-1">{{ alarm.message }}</p>
              </div>
              <div v-if="store.alarms.length === 0" class="py-10 text-center text-slate-600 text-xs italic">No Critical Alarms.</div>
            </div>
          </div>
        </div>

        <div :class="['px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider', 
                      store.backendStatus === 'online' ? 'bg-emerald-500/20 text-emerald-400' : 'bg-rose-500/20 text-rose-400']">
          Server: {{ store.backendStatus }}
        </div>
      </div>
    </header>

    <!-- Tab Navigation -->
    <nav class="flex gap-2 mb-8 bg-slate-800/50 p-1 rounded-xl border border-slate-700 w-fit">
      <router-link to="/" 
                   class="px-6 py-2 rounded-lg text-sm font-bold transition-all flex items-center gap-2"
                   :class="[route.path === '/' ? 'bg-blue-600 text-white shadow-lg' : 'text-slate-400 hover:text-slate-200 hover:bg-slate-700']">
        <svg xmlns="http://www.w3.org/2000/svg" class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7"></rect><rect x="14" y="3" width="7" height="7"></rect><rect x="14" y="14" width="7" height="7"></rect><rect x="3" y="14" width="7" height="7"></rect></svg>
        Dashboard
      </router-link>
      <router-link to="/models" 
                   class="px-6 py-2 rounded-lg text-sm font-bold transition-all flex items-center gap-2"
                   :class="[route.path === '/models' ? 'bg-blue-600 text-white shadow-lg' : 'text-slate-400 hover:text-slate-200 hover:bg-slate-700']">
        <svg xmlns="http://www.w3.org/2000/svg" class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v10"></path><path d="M18.4 4.6a10 10 0 1 1-12.8 0"></path></svg>
        Model Management
      </router-link>
    </nav>

    <main class="flex-1">
      <router-view v-slot="{ Component }">
        <transition name="fade" mode="out-in">
          <component :is="Component" />
        </transition>
      </router-view>
    </main>

    <!-- Floating Toasts -->
    <div class="fixed top-6 right-6 z-[200] flex flex-col gap-3 w-80 pointer-events-none">
      <div v-for="toast in activeToasts" :key="toast.id" 
           :class="['bg-slate-800 border-l-4 p-4 rounded-lg shadow-2xl pointer-events-auto animate-in slide-in-from-right duration-300',
                    toast.type === 'ALARM' ? 'border-rose-500' : 'border-emerald-500']">
        <div class="flex justify-between items-start mb-2">
          <div class="flex items-center gap-2">
            <span v-if="toast.type === 'ALARM'" class="relative flex h-2 w-2">
              <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-rose-400 opacity-75"></span>
              <span class="relative inline-flex rounded-full h-2 w-2 bg-rose-500"></span>
            </span>
            <span v-else class="flex h-2 w-2 rounded-full bg-emerald-500"></span>
            <span :class="['text-xs font-bold uppercase tracking-tighter', 
                           toast.type === 'ALARM' ? 'text-rose-400' : 'text-emerald-400']">
              {{ toast.type === 'ALARM' ? 'Critical Alarm' : 'Alarm Cleared' }}
            </span>
          </div>
          <button @click="activeToasts = activeToasts.filter(t => t.id !== toast.id)" class="text-slate-500 hover:text-slate-300">
            <svg xmlns="http://www.w3.org/2000/svg" class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
          </button>
        </div>
        <div class="text-sm font-bold text-slate-200">{{ toast.ip_addr }} (S{{ toast.slot_id }}/P{{ toast.port_id }})</div>
        <div class="text-xs text-slate-400 mt-1 line-clamp-2">{{ toast.message }}</div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import { useRoute } from 'vue-router'
import { store } from './store'

const route = useRoute()
const activeToasts = ref<any[]>([])
const showNotificationCenter = ref(false)
const sseSource = ref<EventSource | null>(null)

const setupSSE = () => {
  if (sseSource.value) sseSource.value.close()
  
  const source = new EventSource('/api/stream/alarms')
  sseSource.value = source
  
  source.addEventListener('alarm', (event) => {
    // unknown 상태(초기화 중)이거나 running일 때 모두 수신 허용
    if (store.schedulerStatus === 'stopped') return
    
    const data = JSON.parse(event.data)
    console.log('[SSE] Alarm received:', data)
    const key = `${data.ip_addr}-${data.slot_id}-${data.port_id}`
    
    if (data.type === 'ALARM') {
      // 1. 중복 제거: 기존 동일 포트 알람이 있다면 삭제 후 최신으로 교체
      store.alarms = store.alarms.filter(a => `${a.ip_addr}-${a.slot_id}-${a.port_id}` !== key)
      store.alarms.unshift(data)
      if (store.alarms.length > 30) store.alarms.pop()
      
      const toastId = Date.now()
      activeToasts.value.push({ id: toastId, type: 'ALARM', ...data })
      setTimeout(() => {
        activeToasts.value = activeToasts.value.filter(t => t.id !== toastId)
      }, 8000)
    } 
    else if (data.type === 'CLEAR') {
      // 2. 자동 해제: 정상으로 돌아온 포트의 알람은 목록에서 삭제
      const wasExisting = store.alarms.some(a => `${a.ip_addr}-${a.slot_id}-${a.port_id}` !== key)
      store.alarms = store.alarms.filter(a => `${a.ip_addr}-${a.slot_id}-${a.port_id}` !== key)
      
      const toastId = Date.now()
      activeToasts.value.push({ id: toastId, type: 'CLEAR', ...data })
      setTimeout(() => {
        activeToasts.value = activeToasts.value.filter(t => t.id !== toastId)
      }, 5000)
    }

    // 전역 스토어의 디바운스된 fetch 호출
    store.debouncedFetch(1000)
  })

  source.onerror = (err) => {
    console.error('SSE error:', err)
    source.close()
    setTimeout(setupSSE, 5000)
  }
}

onMounted(() => {
  store.fetchAnomalies()
  store.fetchWatchlist()
  store.fetchSchedulerStatus()
  store.fetchActiveAlarms()
  setupSSE()
  
  const interval = setInterval(() => {
    store.fetchAnomalies()
    store.fetchWatchlist()
    store.fetchSchedulerStatus()
  }, 60000)
  onUnmounted(() => clearInterval(interval))
})

onUnmounted(() => {
  if (sseSource.value) sseSource.value.close()
})
</script>

<style>
@keyframes slide-in {
  from { transform: translateX(100%); opacity: 0; }
  to { transform: translateX(0); opacity: 1; }
}
.animate-in {
  animation: slide-in 0.3s ease-out forwards;
}

.fade-in { animation: fadeIn 0.2s ease-out; }
.zoom-in { animation: zoomIn 0.2s ease-out; }

@keyframes fadeIn {
  from { opacity: 0; }
  to { opacity: 1; }
}
@keyframes zoomIn {
  from { opacity: 0; transform: scale(0.95); }
  to { opacity: 1; transform: scale(1); }
}

/* Route Transitions */
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.2s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>
