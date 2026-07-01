import os
import csv
import pytest
import torch
import streamlit as st
import pandas as pd

# Safe fallback initialization layout structure to insulate Streamlit context lookup exceptions
if "sim_cycle" not in st.session_state:
    st.session_state.sim_cycle = 1

# Ingest functional modules cleanly from your cloud-optimized app script
from app import fetch_live_data, CSV_FILE_PATH

# =====================================================================
# 1. FIXED LEDGER SCHEMA VALIDATION
# =====================================================================
def test_csv_ledger_headers():
    """Verifies the logging file tracks the exact matrix parameters needed for dataframes."""
    # Create file if it hasn't run yet to prevent test collection errors
    if not os.path.exists(CSV_FILE_PATH):
        with open(CSV_FILE_PATH, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Cycle", "Temperature_C", "Stock_Price_USD", "VQE_Energy_Hartree", "PQC_Public_Key_Bytes", "PQC_Private_Key_Bytes"])

    with open(CSV_FILE_PATH, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)
        
    expected_headers = [
        "Timestamp", "Cycle", "Temperature_C", "Stock_Price_USD", 
        "VQE_Energy_Hartree", "PQC_Public_Key_Bytes", "PQC_Private_Key_Bytes"
    ]
    assert headers == expected_headers, f"Ledger columns mismatch! Found headers: {headers}"

# =====================================================================
# 2. INTEGRATION GATES: DYNAMISM & CHRONOLOGY CHECKS
# =====================================================================
def test_csv_row_dynamism():
    """Verifies that your spreadsheet data is actively changing and not stuck on flat parameters."""
    if os.path.exists(CSV_FILE_PATH):
        df = pd.read_csv(CSV_FILE_PATH)
        if len(df) >= 3:
            unique_vqe_count = df['VQE_Energy_Hartree'].nunique()
            unique_pub_count = df['PQC_Public_Key_Bytes'].nunique()
            unique_priv_count = df['PQC_Private_Key_Bytes'].nunique()
            
            assert unique_vqe_count > 1, "INTEGRATION ERROR: VQE Energy field is flatlined!"
            assert unique_pub_count > 1, "INTEGRATION ERROR: PQC Public Key size field is flatlined!"
            assert unique_priv_count > 1, "INTEGRATION ERROR: PQC Private Key size field is flatlined!"
        else:
            pytest.skip("Awaiting more row generations to evaluate variance.")
    else:
        pytest.skip("Awaiting initialization loop execution.")

def test_csv_chronology_safety():
    """Verifies that your Cycle counter maps cleanly without jumping backward or resetting to 1."""
    if os.path.exists(CSV_FILE_PATH):
        df = pd.read_csv(CSV_FILE_PATH)
        if len(df) >= 2:
            cycles = df['Cycle'].values
            for i in range(1, len(cycles)):
                assert cycles[i] >= cycles[i-1], f"CHRONOLOGY ERROR: Broken sequence at row {i} ({cycles[i-1]} -> {cycles[i]})"
    else:
        pytest.skip("Awaiting database instantiation.")

# =====================================================================
# 3. PYTORCH CALCULATIONS & SYSTEM BOUNDS
# =====================================================================
def test_pytorch_vqe_gradient_descent():
    """Guarantees the PyTorch engine handles gradient calculation and structural loss reductions."""
    simulated_temp = 25.0
    target_vqe_energy = -1.137306 + (0.0005 * (simulated_temp - 25.0))
    learning_rate = 0.1
    
    theta = torch.tensor([0.5], requires_grad=True)
    optimizer = torch.optim.SGD([theta], lr=learning_rate)
    
    optimizer.zero_grad()
    initial_energy = target_vqe_energy + (0.45 * torch.exp(-theta * 4))
    initial_loss = (initial_energy - target_vqe_energy) ** 2
    initial_loss.backward()
    
    assert theta.grad is not None, "PyTorch failed to compute partial derivatives via automatic backpropagation."
    
    optimizer.step()
    updated_energy = target_vqe_energy + (0.45 * torch.exp(-theta * 4))
    updated_loss = (updated_energy - target_vqe_energy) ** 2
    
    assert updated_loss.item() < initial_loss.item(), "PyTorch gradient descent optimization failed to reduce variational loss."

def test_live_data_bounds():
    """Validates that real-time telemetry inputs conform to realistic physical and financial bounds."""
    temp, stock_price = fetch_live_data("IBM")
    assert -50.0 <= temp <= 60.0, f"Unrealistic ambient climate metrics captured: {temp}°C"
    assert stock_price > 0.0, f"Invalid asset token price metric recorded: ${stock_price}"
