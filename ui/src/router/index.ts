import { createRouter, createWebHistory } from 'vue-router'
import DashboardView from '../views/DashboardView.vue'
import ModelManagementView from '../views/ModelManagementView.vue'
import RuleManagementView from '../views/RuleManagementView.vue'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/',
      name: 'dashboard',
      component: DashboardView
    },
    {
      path: '/models',
      name: 'models',
      component: ModelManagementView
    },
    {
      path: '/rules',
      name: 'rules',
      component: RuleManagementView
    }
  ]
})

export default router
