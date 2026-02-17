
import streamlit as st
import numpy as np
import pandas as pd
import pulp
import altair as alt

# --- Configuration & Constants ---
OVERSEAS_COST = 5.0
LOCAL_COST = 6.5
HOLDING_COST = 1.0
BACKORDER_COST = 2.5
LOCAL_LEAD_TIME = 1
OVERSEAS_LEAD_TIME = 3
LOCAL_CAPACITY = 500
DISRUPTION_CHANCE = 0.10
DISRUPTION_DURATION = 3
DEMAND_MEAN_NORMAL = 600
DEMAND_MEAN_DISRUPTED = 1200
DEMAND_STD = 250

st.set_page_config(page_title="Supply Chain Support", layout="wide")

# --- Scenario Generation ---
def generate_scenarios(n_scenarios, horizon, start_disruption_weeks, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    scenarios = []
    
    for _ in range(n_scenarios):
        # State for this scenario
        disruption_remaining = start_disruption_weeks
        
        scenario_demands = []
        scenario_disruptions = [] # Status at start of week t
        scenario_overseas_available = []
        
        for t in range(horizon):
            # Check disruption status at start of week t
            is_disrupted = disruption_remaining > 0
            
            # Record availability for ordering (Overseas unavailable if disrupted)
            scenario_overseas_available.append(not is_disrupted)
            scenario_disruptions.append(is_disrupted)
            
            # Generate demand for week t
            mean = DEMAND_MEAN_DISRUPTED if is_disrupted else DEMAND_MEAN_NORMAL
            demand = max(0, np.random.normal(mean, DEMAND_STD)) # Truncated normal
            scenario_demands.append(demand)
            
            # Evolve disruption state for NEXT week
            if disruption_remaining > 0:
                disruption_remaining -= 1
            else:
                # 10% chance to start disruption (starts effectively next week)
                if np.random.random() < DISRUPTION_CHANCE:
                    disruption_remaining = DISRUPTION_DURATION
        
        scenarios.append({
            "demand": scenario_demands,
            "overseas_available": scenario_overseas_available,
            "disrupted": scenario_disruptions
        })
    
    return scenarios

# --- Optimization Model ---
def solve_optimization(
    current_inventory, 
    current_backlog, 
    pipe_local_next, # Arriving at T+1 (from past)
    pipe_os_1,       # Arriving at T+1 (from past)
    pipe_os_2,       # Arriving at T+2 (from past)
    pipe_os_3,       # Arriving at T+3 (from past ?)
    scenarios,
    horizon,
    local_enabled,
    overseas_enabled
):
    # PuLP Problem
    prob = pulp.LpProblem("SupplyChainOptimization", pulp.LpMinimize)
    
    # --- Decision Variables ---
    # Global variables for T=0 (Implementing Non-Anticipativity)
    local_order_0 = pulp.LpVariable("Local_Order_0", lowBound=0, upBound=LOCAL_CAPACITY if local_enabled else 0)
    overseas_order_0 = pulp.LpVariable("Overseas_Order_0", lowBound=0)
    
    # Check T=0 constraints based on initial state
    if not scenarios[0]['overseas_available'][0] or not overseas_enabled:
         prob += overseas_order_0 == 0
    
    # Variables per scenario
    inv_vars = {}
    back_vars = {}
    local_vars = {}
    overseas_vars = {}
    
    N = len(scenarios)
    total_cost_expr = 0
    
    for s_idx, scenario in enumerate(scenarios):
        # --- Helpers to get order variables at time t ---
        def get_local_order(t):
            if t == 0: return local_order_0
            return local_vars[(t, s_idx)]

        def get_overseas_order(t):
            if t == 0: return overseas_order_0
            return overseas_vars[(t, s_idx)]
        
        # Create variables for this scenario
        for t in range(1, horizon):
            l_t = pulp.LpVariable(f"Loc_{t}_{s_idx}", 0, LOCAL_CAPACITY if local_enabled else 0)
            o_t = pulp.LpVariable(f"Ovr_{t}_{s_idx}", 0)
            
            if not scenario['overseas_available'][t] or not overseas_enabled:
                prob += o_t == 0
                
            local_vars[(t, s_idx)] = l_t
            overseas_vars[(t, s_idx)] = o_t
            
        # --- Inventory Balance & Arrivals ---
        prev_inv_expr = current_inventory - current_backlog # State at start of T=0
        scenario_cost = 0
        
        # Cost of T=0 orders
        scenario_cost += LOCAL_COST * local_order_0
        scenario_cost += OVERSEAS_COST * overseas_order_0
        
        for t in range(horizon):
            inv_t = pulp.LpVariable(f"Inv_{t}_{s_idx}", 0)
            back_t = pulp.LpVariable(f"Back_{t}_{s_idx}", 0)
            inv_vars[(t, s_idx)] = inv_t
            back_vars[(t, s_idx)] = back_t
            
            # Calculate Arrivals at T
            arrivals = 0
            if t == 0:
                arrivals = 0
            elif t == 1:
                arrivals = pipe_local_next + pipe_os_1 + get_local_order(0)
            elif t == 2:
                arrivals = pipe_os_2 + get_local_order(1)
            elif t == 3:
                arrivals = pipe_os_3 + get_local_order(2) + get_overseas_order(0)
            else:
                arrivals = get_local_order(t-1) + get_overseas_order(t-3)
            
            # Balance Constraint
            if t == 0:
                starts_expr = prev_inv_expr
            else:
                starts_expr = inv_vars[(t-1, s_idx)] - back_vars[(t-1, s_idx)]
                
            prob += inv_t - back_t == starts_expr + arrivals - scenario['demand'][t]
            
            # Costs
            scenario_cost += HOLDING_COST * inv_t + BACKORDER_COST * back_t
            
            # Order costs for future weeks
            if t > 0:
                scenario_cost += LOCAL_COST * get_local_order(t)
                scenario_cost += OVERSEAS_COST * get_overseas_order(t)
        
        total_cost_expr += scenario_cost

    prob += total_cost_expr * (1/N)
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    rec_local = pulp.value(local_order_0)
    rec_overseas = pulp.value(overseas_order_0)
    
    # Extract Metrics
    sim_data = [] 
    scenario_costs = []
    
    for s in range(N):
        c = 0
        c += LOCAL_COST * rec_local + OVERSEAS_COST * rec_overseas
        c += HOLDING_COST * pulp.value(inv_vars[(0,s)]) + BACKORDER_COST * pulp.value(back_vars[(0,s)])
        for t in range(1, horizon):
            c += LOCAL_COST * pulp.value(local_vars[(t,s)]) + OVERSEAS_COST * pulp.value(overseas_vars[(t,s)])
            c += HOLDING_COST * pulp.value(inv_vars[(t,s)]) + BACKORDER_COST * pulp.value(back_vars[(t,s)])
        scenario_costs.append(c)
        
        for t in range(horizon):
            net_stock = pulp.value(inv_vars[(t,s)]) - pulp.value(back_vars[(t,s)])
            sim_data.append({
                "Scenario": s,
                "Week": t,
                "Net Inventory": net_stock
            })
            
    df_sim = pd.DataFrame(sim_data)
    df_costs = pd.DataFrame({"Cost": scenario_costs})
    
    backorder_count_t1 = sum(1 for s in range(N) if pulp.value(back_vars[(1,s)]) > 0.1)
    prob_backorder_t1 = backorder_count_t1 / N
    exp_inv_t1 = sum((pulp.value(inv_vars[(1,s)]) - pulp.value(back_vars[(1,s)])) for s in range(N)) / N
    
    return {
        "local_order": rec_local,
        "overseas_order": rec_overseas,
        "prob_backorder_t1": prob_backorder_t1,
        "expected_inv_t1": exp_inv_t1,
        "df_sim": df_sim,
        "df_costs": df_costs
    }

# --- Pages ---

def show_logic():
    st.title("🧠 Model Logic & Assumptions")
    
    st.markdown("""
    ### 1. The Optimization Approach
    This tool uses **Two-Stage Stochastic Programming** to make the best decision for the *current* week, considering uncertain future events.
    
    #### Why "300 Futures"?
    We simulate 300 independent timelines (scenarios) because the future is uncertain in **two ways**:
    
    1.  **variable Demand**: In every week of every scenario, demand is random (drawn from a Bell curve).
        *   *Scenario 1* might have consistently high demand.
        *   *Scenario 2* might have low demand.
    2.  **Random Disruption**: In every week, there is a **10% chance** of a disruption starting.
        *   *Scenario A*: No disruption ever occurs (Smooth sailing).
        *   *Scenario B*: Disruption hits immediately in Week 1.
        *   *Scenario C*: Disruption hits late in Week 6.
    
    By optimizing across all 300 combined possibilities, we find an order quantity that is **robust**—it minimizes your *average* cost whether you get lucky (no disruption) or unlucky.
    
    ### 2. Game Parameters & Costs
    The model is hard-coded with the following game rules:
    
    | Parameter | Value | Notes |
    | :--- | :--- | :--- |
    | **Local Cost** | $6.5 / unit | Expensive, but fast. |
    | **Overseas Cost** | $5.0 / unit | Cheap, but slow. |
    | **Holding Cost** | $1.0 / unit / week | Charged on inventory at end of week. |
    | **Backorder Cost** | $2.5 / unit / week | Charged on backlog at end of week. |
    | **Local Lead Time** | 1 Week | Order in Week T \u2192 Arrives Start of T+1. |
    | **Overseas Lead Time** | 3 Weeks | Order in Week T \u2192 Arrives Start of T+3. |
    
    ### 3. Disruption Logic
    *   **Normal State**: Demand ~ N(600, 250). Overseas is available.
    *   **Disruption**: 
        *   10% chance to start each week if not already disrupted.
        *   Lasts exactly **3 weeks**.
        *   **Overseas Supplier** becomes **unavailable** (Capacity = 0).
        *   **Demand** doubles to ~ N(1200, 250).
    
    ### 4. Mathematical Goal
    The solver minimizes the objective function:
    
    $$
    \\min E \\left[ \\sum_{t=0}^{H} (6.5 x_{local,t} + 5.0 x_{overseas,t} + 1.0 I_t + 2.5 B_t) \\right]
    $$
    
    Where:
    *   $x$ = Order quantities
    *   $I$ = Inventory held
    *   $B$ = Backlog
    *   The expectation $E[...]$ is the average over 300 scenarios.
    """)

def show_optimizer():
    st.title("📦 Supply Chain Decision Support")

    st.markdown("""
    ### 🎯 Objective
    Minimize total costs over the next 8 weeks.
    *   **Holding Cost**: $1.0 / unit / week (avoid excess inventory)
    *   **Backorder Cost**: $2.5 / unit / week (avoid stockouts)

    ### 📝 Instructions
    1.  **Enter Current Status**: Input your current inventory and backlog from the game.
    2.  **Update Pipeline**: Enter orders *already placed* that will arrive in T+1, T+2, T+3.
    3.  **Check Disruption**: Set the disruption status if active.
    4.  **Run Optimizer**: The tool will simulate 300 futures to find the best order for *this week*.
    """)

    with st.sidebar:
        st.header("1. Game Parameters")
        
        # Dynamic Game Length Logic
        final_week = 20 # Default
        if st.session_state.get('current_week_input', 1) > 20:
            final_week = 30
            st.warning("⚠️ Game extended to 30 weeks!")
            
        settings_horizon = st.number_input(
            "Max Lookahead (Weeks)", 4, 12, 8,
            help="How many weeks ahead the model optimizes."
        )
        
        n_scenarios = st.number_input(
            "Scenarios", 10, 1000, 300,
            help="Number of random futures to simulate. Higher = more accurate but slower."
        )
        seed = st.number_input("Random Seed", 0, 9999, 42)
        
        st.header("2. Current State (Week T)")
        
        st.subheader("Inventory Status")
        st.caption("Input the state at the *start* of the current week.")
        col1, col2 = st.columns(2)
        current_week = col1.number_input("Current Week #", 1, 50, 1, key='current_week_input')
        on_hand = col1.number_input("On-hand Inventory", 0, 10000, 1000)
        backlog = col2.number_input("Backorders", 0, 10000, 0)
        
        # Calculate Effective Horizon
        weeks_remaining = final_week - current_week + 1
        effective_horizon = min(settings_horizon, weeks_remaining)
        
        # Determine if we apply Terminal Value
        # Apply it unless we are literally at the end of the game (lookahead reaches final week)
        # If weeks_remaining <= settings_horizon, we are seeing the "End of the World".
        # If we see the end, we SHOULDN'T apply terminal value? 
        # Actually, if we are at Week 19 of 20, we want to end Week 20 with 0.
        # So: Apply Terminal Value ONLY if weeks_remaining > settings_horizon.
        # i.e., The game continues BEYOND what we can see.
        apply_terminal = weeks_remaining > settings_horizon
        
        st.info(f" optimizing for {effective_horizon} weeks. (End: W{current_week + effective_horizon - 1})")
        if apply_terminal:
            st.success("🔄 Infinite Horizon Mode (Holding stock for future)")
        else:
            st.warning("🏁 End Game Mode (Draining inventory)")

        st.subheader("Pipeline (Incoming)")
        st.caption("Enter quantities ALREADY ordered that are arriving soon.")
        pipe_local = st.number_input(
            "Local Arriving Next Week (T+1)", 0, 5000, 0,
            help="Order placed last week (T-1). Arrives start of next week."
        )
        pipe_os1 = st.number_input(
            "Overseas Arriving in 1 wk (T+1)", 0, 5000, 0,
            help="Order placed 3 weeks ago. Arrives start of next week."
        )
        pipe_os2 = st.number_input(
            "Overseas Arriving in 2 wks (T+2)", 0, 5000, 0,
            help="Order placed 2 weeks ago."
        )
        pipe_os3 = st.number_input(
            "Overseas Arriving in 3 wks (T+3)", 0, 5000, 0,
            help="Order placed last week."
        )
        
        st.subheader("Disruption Status")
        disruption_rem = st.slider(
            "Disruption Remaining Weeks", 0, 3, 0, 
            help="0 = Normal. 1-3 = Currently disrupted. Overseas will be unavailable."
        )
        
        st.subheader("Supplier Setup")
        local_on = st.checkbox("Local Supplier Enabled", True)
        overseas_on = st.checkbox("Overseas Supplier Enabled", True)
        
        run_btn = st.button("🚀 Run Optimizer", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner("Generating scenarios and optimizing..."):
            # 1. Generate Scenarios
            scenarios = generate_scenarios(n_scenarios, effective_horizon, disruption_rem, seed)
            
            # 2. Solve
            res = solve_optimization(
                on_hand, backlog, 
                pipe_local, pipe_os1, pipe_os2, pipe_os3,
                scenarios, effective_horizon, local_on, overseas_on,
                apply_terminal_value=apply_terminal
            )
            
            st.session_state['res'] = res

    if 'res' in st.session_state:
        res = st.session_state['res']
        
        # --- Main Recommendation ---
        st.markdown("---")
        st.subheader(f"✅ Recommendations for Week {current_week}")
        st.caption("Enter these orders into the game *now*.")
        
        col_rec1, col_rec2 = st.columns(2)
        
        with col_rec1:
            st.info("**Local Order** (Cost 6.5, Lead 1 wk)")
            st.metric("Quantity", f"{res['local_order']:,.0f}", delta=None)
            st.caption("Quick but expensive. Use when backlog triggers are high.")
            
        with col_rec2:
            st.info("**Overseas Order** (Cost 5.0, Lead 3 wks)")
            st.metric("Quantity", f"{res['overseas_order']:,.0f}", delta=None)
            st.caption("Cheap but slow. Main source of stock.")
        
        # --- Risk Analysis ---
        st.markdown("### 📊 Predicted Outcomes (End of Week T+1)")
        st.caption("Projected status *after* current decisions takes effect (Next Week).")
        r1, r2, r3 = st.columns(3)
        r1.metric(
            "Backorder Risk", 
            f"{res['prob_backorder_t1']*100:.1f}%",
            help="Probability of having ANY backorders next week."
        )
        r2.metric(
            "Exp. Net Inventory", 
            f"{res['expected_inv_t1']:,.0f}",
            help="Average projected inventory (Inventory - Backlog)."
        )
        r3.metric(
            "Avg Scenario Cost", 
            f"${res['df_costs']['Cost'].mean():,.0f}",
            help="Average total cost over the 8-week horizon."
        )
        
        # --- Visualizations ---
        tab1, tab2 = st.tabs(["Inventory Projection", "Cost Distribution"])
        
        with tab1:
            st.markdown("#### Inventory Fan Chart (Net Inventory)")
            st.info(
                """
                **How to read:**
                *   **Blue Line (Middle)**: Most likely inventory level.
                *   **Shaded Area**: Range of outcomes (10th to 90th percentile).
                *   **Risk**: If the shaded area drops below 0, there is a risk of backlog.
                """
            )
            # Aggregate stats by week
            df = res['df_sim']
            stats = df.groupby("Week")["Net Inventory"].quantile([0.1, 0.5, 0.9]).unstack()
            stats.columns = ["P10", "P50", "P90"]
            stats = stats.reset_index()
            
            # Transform for Altair
            base = alt.Chart(stats).encode(x="Week:O")
            
            # Area for P10-P90
            area = base.mark_area(opacity=0.3, color='blue').encode(
                y='P10', 
                y2='P90'
            )
            
            # Line for Median
            line = base.mark_line(color='blue').encode(
                y='P50'
            )
            
            st.altair_chart((area + line).interactive(), use_container_width=True)
        
        with tab2:
            st.markdown("#### Cost Distribution across Scenarios")
            st.info(
                """
                **How to read:**
                *   **Bars**: Show how many scenarios result in a specific total cost.
                *   **Goal**: You want tall bars on the *left* (low cost).
                *   **Tail**: Bars far to the *right* show "disaster" scenarios (high cost).
                """
            )
            chart = alt.Chart(res['df_costs']).mark_bar().encode(
                x=alt.X("Cost", bin=True),
                y='count()'
            )
            st.altair_chart(chart, use_container_width=True)

# --- Navigation ---
page = st.sidebar.radio("Navigation", ["Decision Support", "Model Logic"])

if page == "Decision Support":
    show_optimizer()
else:
    show_logic()
