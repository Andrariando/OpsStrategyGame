
import streamlit as st
import numpy as np
import pandas as pd
import pulp
import altair as alt
import scipy.stats as stats

# --- Configuration & Constants ---
# --- Configuration & Constants ---
OVERSEAS_COST = 5.0
LOCAL_COST = 7.5      # Increased from 6.5
HOLDING_COST = 1.0
BACKORDER_COST = 3.0  # Increased from 2.5
LOCAL_LEAD_TIME = 1
OVERSEAS_LEAD_TIME = 4 # Increased from 3
LOCAL_CAPACITY = 500
DISRUPTION_CHANCE = 0.10
DISRUPTION_DURATION = 5 # Increased from 3
DEMAND_MEAN_NORMAL = 500 # Changed from 600
DEMAND_MEAN_DISRUPTED = 1200
DEMAND_STD = 350      # Changed from 250

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

# --- Base Stock Policy Benchmark ---

def calculate_optimal_base_stock():
    """
    Calculates the theoretical Base Stock level (B) based on the Critical Ratio.
    Policy assumes Normal Demand and Overseas Supplier (Primary).
    """
    # Critical Ratio (Service Level Target)
    # Cu = Backorder Cost, Co = Holding Cost
    # CR = Cu / (Cu + Co)
    critical_ratio = BACKORDER_COST / (BACKORDER_COST + HOLDING_COST)
    
    # Lead Time + Review Period
    # We review every week (r=1). Lead time L=3.
    L = OVERSEAS_LEAD_TIME
    r = 1
    review_plus_lead = r + L
    
    # Demand during coverage period
    mu_L = DEMAND_MEAN_NORMAL * review_plus_lead
    sigma_L = DEMAND_STD * np.sqrt(review_plus_lead)
    
    # Z-score for Critical Ratio
    z = stats.norm.ppf(critical_ratio)
    
    # Base Stock Formula: B = mu_L + z * sigma_L
    base_stock = mu_L + z * sigma_L
    
    return base_stock, critical_ratio

def simulate_base_stock_policy(
    base_stock_level,
    current_inventory,
    current_backlog,
    pipe_local_next,
    pipe_os_1, pipe_os_2, pipe_os_3,
    scenarios,
    horizon,
    local_enabled,
    overseas_enabled
):
    """
    Simulates a 'Naive' Base Stock policy over the same scenarios.
    It purely uses the Overseas supplier to perform Order-Up-To B.
    It ignores future disruption risks (reactive).
    """
    results = []
    
    for s_idx, scenario in enumerate(scenarios):
        # Initialize State for this scenario path
        inv = current_inventory
        back = current_backlog
        
        # Pipeline orders (tracked by arrival time)
        # arrivals[0] = arriving T (now/processing), arrivals[1] = T+1, etc.
        # We model pipeline simply as a list of incoming orders.
        # Initial Pipeline:
        # T+1: pipe_local_next + pipe_os_1
        # T+2: pipe_os_2
        # T+3: pipe_os_3
        # T+4+: 0
        
        pipeline = {
            1: pipe_local_next + pipe_os_1,
            2: pipe_os_2,
            3: pipe_os_3
        }
        
        total_cost = 0
        scenario_history = []
        
        for t in range(horizon):
            # 1. Determine Inventory Position (IP)
            # IP = Net Inventory + On Order (arriving T+1 onwards)
            # Pipeline keys > t represent future arrivals relative to NOW (t=0)
            # But inside the loop, 't' moves.
            # arrivals at 't' are consumed.
            # arrivals > 't' are on order.
            
            # Correction: Dictionary 'pipeline' stores ABSOLUTE arrival times (0, 1, 2...).
            # 'pulp' model treats t=0 as "Decide Now".
            # arrivals at t=1, t=2... are fixed.
            # New orders placed at 't' arrive at 't+L'.
            
            on_order = sum(qty for arr_t, qty in pipeline.items() if arr_t > t)
            net_inv = inv - back
            ip = net_inv + on_order
            
            # 2. Place Order (Review Period)
            # Order up to Base Stock Level
            # If disrupted or overseas disabled, we CANNOT order overseas.
            
            # Theoretical Policy Target: Order = Max(0, B - IP)
            raw_order = max(0, base_stock_level - ip)
            
            # Apply Constraints
            # In this 'Simple' policy, we assume we want to use the Cheap Overseas supplier.
            
            actual_order_os = 0
            actual_order_loc = 0
            
            # Check availability at time 't'
            # Note: scenario['overseas_available'] is a list of bools
            if overseas_enabled and scenario['overseas_available'][t]:
                actual_order_os = raw_order
            elif local_enabled:
                # Fallback to local if primary is down (Simple Managers Heuristic)
                # But capped at capacity
                actual_order_loc = min(raw_order, LOCAL_CAPACITY)
            
            # Record Order Cost
            total_cost += (actual_order_os * OVERSEAS_COST) + (actual_order_loc * LOCAL_COST)
            
            # Add to Pipeline
            # Local arrives T+1
            if actual_order_loc > 0:
                arr_t = t + 1
                pipeline[arr_t] = pipeline.get(arr_t, 0) + actual_order_loc
            
            # Overseas arrives T+3
            if actual_order_os > 0:
                arr_t = t + 3
                pipeline[arr_t] = pipeline.get(arr_t, 0) + actual_order_os
            
            # 3. Receive Arrivals (Start of week/During week)
            # At start of week t, orders scheduled for t arrive.
            arrivals = pipeline.get(t, 0)
            
            # 4. Satisfy Demand
            demand = scenario['demand'][t]
            
            # Balance
            # Start Inv = Prev End Inv
            # Available = Start + Arrivals
            available = (inv - back) + arrivals
            
            if available >= demand:
                inv = available - demand
                back = 0
            else:
                back = demand - available
                inv = 0
            
            # 5. Cost
            total_cost += (inv * HOLDING_COST) + (back * BACKORDER_COST)
            
            scenario_history.append({
                "Week": t,
                "Net Inventory": inv - back,
                "Policy": "Base Stock"
            })
            
        results.append({
            "Cost": total_cost,
            "History": scenario_history
        })
        
    return results

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
    overseas_enabled,
    apply_terminal_value=False
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

        # --- Terminal Value (End of Horizon) ---
        # If we are in "Mid-Game" (apply_terminal_value=True), we must not drain inventory.
        # We credit the ending inventory to avoid "end of world" behavior.
        # Valuation: We value it at OVERSEAS_COST (conservative replacement cost).
        if apply_terminal_value:
            # Credit Ending Inventory: - (Inv_T * Value)
            # We subtract this from the Cost to minimize.
            # This encourages the solver to leave inventory > 0 if it's useful.
            # Note: We do NOT credit pipeline, as that's already paid for? 
            # actually, standard practice is just to ensure we don't dump.
            # Let's subtract (Ending Inventory * OVERSEAS_COST) from the objective.
            
            end_inv = inv_vars[(horizon-1, s_idx)]
            total_cost_expr -= end_inv * OVERSEAS_COST

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
        st.header("1. Game Status")
        
        col1, col2 = st.columns(2)
        current_week = col1.number_input("Current Week #", 1, 50, 1)

        # Default settings (hidden by default)
        with st.expander("⚙️ Advanced Model Settings"):
            settings_horizon = st.number_input("Max Lookahead", 4, 12, 8)
            n_scenarios = st.number_input("Scenarios", 10, 1000, 300)
            seed = st.number_input("Random Seed", 0, 9999, 42)

        # --- Game Phase & Week ---
        # Allow manual override of the "Max Weeks"
        game_length_option = st.radio(
            "Game Duration", 
            ["Standard (20 Weeks)", "Extended (30 Weeks)", "Infinite (Indefinite)"],
            horizontal=True,
            help="Select the total number of weeks in the game."
        )
        
        if "Infinite" in game_length_option:
            final_week = 999 # Effectively infinite
            apply_terminal = True
            effective_horizon = settings_horizon
            st.success(f"♾️ **Infinite Horizon Strategy**\n\nOptimizing for **{effective_horizon} weeks** ahead.\n\nAssumes the game goes on forever (Steady State).")
        else:
            final_week = 20 if "20" in game_length_option else 30
            # Calculate Effective Horizon
            weeks_remaining = final_week - current_week + 1
            effective_horizon = min(settings_horizon, weeks_remaining)
            apply_terminal = weeks_remaining > settings_horizon
            
            if apply_terminal:
                st.info(f"🟢 **Mid-Game Strategy**\n\nOptimizing for **{effective_horizon} weeks** ahead.")
            else:
                st.warning(f"🏁 **End-Game Strategy**\n\nOptimizing for final **{weeks_remaining} weeks**.\n\nDraining inventory.")

        st.divider()
        st.header("2. Current Inventory")
        st.caption("Input start-of-week status.")
        
        col_inv, col_back = st.columns(2)
        on_hand = col_inv.number_input("On-hand", 0, 10000, 2500)
        backlog = col_back.number_input("Backorders", 0, 10000, 0)
        
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
            
            # 2. Solve Optimization (The "AI")
            res_opt = solve_optimization(
                on_hand, backlog, 
                pipe_local, pipe_os1, pipe_os2, pipe_os3,
                scenarios, effective_horizon, local_on, overseas_on
            )
            
            # 3. Simulate Base Stock (The "Benchmark")
            optimal_B, critical_ratio = calculate_optimal_base_stock()
            res_bs = simulate_base_stock_policy(
                optimal_B,
                on_hand, backlog,
                pipe_local, pipe_os1, pipe_os2, pipe_os3,
                scenarios, effective_horizon, local_on, overseas_on
            )
            
            # Process Benchmark Results
            bs_costs = [r['Cost'] for r in res_bs]
            bs_history = [item for r in res_bs for item in r['History']]
            
            df_costs_bs = pd.DataFrame({"Cost": bs_costs, "Policy": "Base Stock"})
            df_sim_bs = pd.DataFrame(bs_history)
            
            # Merge
            df_costs_all = pd.concat([res_opt['df_costs'], df_costs_bs])
            df_sim_all = pd.concat([res_opt['df_sim'], df_sim_bs])

            st.session_state['res'] = {
                'opt': res_opt,
                'bs': {
                    'B': optimal_B,
                    'CR': critical_ratio,
                    'df_costs': df_costs_bs,
                    'df_sim': df_sim_bs
                },
                'combined': {
                    'df_costs': df_costs_all,
                    'df_sim': df_sim_all
                }
            }

    if 'res' in st.session_state:
        res = st.session_state['res']
        res_opt = res['opt']
        res_bs = res['bs']
        
        # --- Main Recommendation ---
        st.markdown("---")
        st.subheader(f"✅ Recommendations for Week {current_week}")
        st.caption("AI-Derived Optimal Orders")
        
        col_rec1, col_rec2, col_stats = st.columns([1, 1, 2])
        
        with col_rec1:
            st.info("**Local Order**")
            st.metric("Qty", f"{res_opt['local_order']:,.0f}")
            st.caption("Lead 1 wk | Cost $6.5")
            
        with col_rec2:
            st.info("**Overseas Order**")
            st.metric("Qty", f"{res_opt['overseas_order']:,.0f}")
            st.caption("Lead 3 wks | Cost $5.0")
            
        with col_stats:
            # Comparison with Tool
            st.warning("📊 **Benchmark Comparison**")
            
            # Calculate what Base Stock Would Order NOW
            # Net Inv + Pipeline
            pipeline_total = pipe_local + pipe_os1 + pipe_os2 + pipe_os3
            current_ip = (on_hand - backlog) + pipeline_total
            bs_order = max(0, res_bs['B'] - current_ip)
            
            c1, c2 = st.columns(2)
            c1.metric("Theoretical Base Stock", f"{res_bs['B']:,.0f}", help=f"Optimal B for {res_bs['CR']:.0%} Service Level")
            c2.metric("Base Stock Order", f"{bs_order:,.0f}", delta=f"{res_opt['local_order'] + res_opt['overseas_order'] - bs_order:,.0f} vs AI", help="Using formula max(0, B - IP)")
            
            if bs_order > (res_opt['local_order'] + res_opt['overseas_order']):
                 st.caption("🤖 AI assumes demand may drop or costs rising.")
            else:
                 st.caption("🤖 AI is stocking up (likely disruption fear).")

        # --- Visualizations ---
        st.markdown("### ⚔️ Strategy Duel: AI vs. Textbook")
        
        tab1, tab2, tab3 = st.tabs(["Cost Comparison", "Inventory Dynamics", "Why is AI Better?"])
        
        with tab1:
            st.markdown("#### Distribution of Total Costs (Lower is Better)")
            st.caption("We simulated 300 futures for both strategies. The AI strategy usually shifts the curve to the left (cheaper).")
            
            # Calculate Averages
            avg_opt = res_opt['df_costs']['Cost'].mean()
            avg_bs = res_bs['df_costs']['Cost'].mean()
            
            c1, c2 = st.columns(2)
            c1.metric("Avg Cost (AI)", f"${avg_opt:,.0f}")
            c2.metric("Avg Cost (Base Stock)", f"${avg_bs:,.0f}", delta=f"${avg_opt - avg_bs:,.0f}", delta_color="inverse")

            chart = alt.Chart(res['combined']['df_costs']).mark_area(
                opacity=0.5,
                interpolate='step'
            ).encode(
                x=alt.X("Cost", bin=alt.Bin(maxbins=30)),
                y=alt.Y('count()', stack=None),
                color='Policy'
            ).properties(height=300)
            
            st.altair_chart(chart, use_container_width=True)
            
        with tab2:
            st.markdown("#### Inventory Levels over Time")
            st.caption("Comparison of Net Inventory behavior. Does the AI keep leaner stock?")
            
            # Aggregate stats by week & Policy
            df_all = res['combined']['df_sim']
            
            # Simpler: standard aggregation
            agg = df_all.groupby(["Week", "Policy"])["Net Inventory"].mean().reset_index()
            
            base = alt.Chart(agg).mark_line(point=True).encode(
                x="Week:O",
                y="Net Inventory",
                color="Policy",
                tooltip=["Week", "Policy", "Net Inventory"]
            ).properties(height=300)
            
            # Add Horizontal Rule for Base Stock B
            rule = alt.Chart(pd.DataFrame({'B': [res_bs['B']]})).mark_rule(color='red', strokeDash=[5,5]).encode(y='B')
            
            st.altair_chart((base + rule).interactive(), use_container_width=True)

        with tab3:
            st.markdown("### 🧠 Logic Explanation")
            st.markdown(f"""
            #### 1. The "Base Stock" Policy
            *   **Formula**: $B = \mu(r+L) + z\sigma\sqrt{{r+L}}$
            *   **Your Inputs**: Analysis of *Standard Normal Demand* requires a Stock Level of **{res_bs['B']:,.0f}**.
            *   **Flaw**: This policy is static. It does not know about the **End of Game** (holding useless stock in week 20) or **Disruptions** (sudden capacity loss).
            
            #### 2. The "Stochastic Optimizer" (AI)
            *   **Optimization**: It explicitly minimizes cost over 300 unique future scenarios.
            *   **Dynamic**: 
                *   If it sees a Disruption Risk, it pre-orders (Panic Buying) \u2192 *Analysis: See if Inventory Chart provides a spike.*
                *   If the game is ending, it stops ordering \u2192 *Analysis: AI Inventory drops to 0 at the end, Base Stock stays high.*
                *   It uses the **Dual Source**, balancing the cheap Overseas option against the fast Local one.
            """)
            st.altair_chart(chart, use_container_width=True)

# --- Navigation ---
page = st.sidebar.radio("Navigation", ["Decision Support", "Model Logic"])

if page == "Decision Support":
    show_optimizer()
else:
    show_logic()
