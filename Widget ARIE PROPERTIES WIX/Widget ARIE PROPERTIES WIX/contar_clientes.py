import requests
import base64
import json
import os

# Configurações
usuario = "arieproperties-ti"
senha = "eeoTCIXR9NtEdv118V1L16xZWZR64W7p"
subdominio = "arieproperties"  # Subdomínio da Arie

credenciais = f"{usuario}:{senha}"
token = base64.b64encode(credenciais.encode()).decode()

headers = {
    "Authorization": f"Basic {token}",
    "Content-Type": "application/json"
}

# 1. Baixar o JSON de boletos/parcelas (ajuste o endpoint conforme necessário)
cpf = "37455407866"
endpoint_boletos = f"https://api.sienge.com.br/{subdominio}/public/api/v1/current-debit-balance?cpf={cpf}"

response_boletos = requests.get(endpoint_boletos, headers=headers)

if response_boletos.status_code == 200:
    boletos_data = response_boletos.json()
    boletos_path = os.path.join(os.path.dirname(__file__), "boletos.json")
    with open(boletos_path, "w", encoding="utf-8") as f:
        json.dump(boletos_data, f, ensure_ascii=False, indent=2)
    print("Arquivo boletos.json criado com sucesso!")
else:
    print(f"Erro ao baixar boletos: {response_boletos.status_code}")
    print(response_boletos.text)
    exit(1)

# 2. Processar o arquivo para buscar a segunda via do boleto
with open(boletos_path, "r", encoding="utf-8") as f:
    data = json.load(f)

bill_receivable_id = None
installment_id = None

for bill in data.get("results", []):
    # Verifica parcelas a vencer e vencidas
    for key in ["dueInstallments", "payableInstallments", "paidInstallments"]:
        for inst in bill.get(key, []):
            if inst.get("generatedBoleto") and inst.get("currentBalance", 0) > 0:
                bill_receivable_id = bill.get("billReceivableId")
                installment_id = inst.get("installmentId")
                break
        if bill_receivable_id and installment_id:
            break
    if bill_receivable_id and installment_id:
        break

if not bill_receivable_id or not installment_id:
    print("Nenhuma parcela apta para segunda via de boleto encontrada.")
else:
    endpoint = f"https://api.sienge.com.br/{subdominio}/public/api/v1/payment-slip-notification?billReceivableId={bill_receivable_id}&installmentId={installment_id}"
    response = requests.get(endpoint, headers=headers)
    if response.status_code == 200:
        slip_data = response.json()
        with open("payment_slip_notification_auto.json", "w", encoding="utf-8") as f:
            json.dump(slip_data, f, ensure_ascii=False, indent=2)
        print(f"Arquivo payment_slip_notification_auto.json criado com sucesso! (billReceivableId={bill_receivable_id}, installmentId={installment_id})")
        print(json.dumps(slip_data, ensure_ascii=False, indent=2))
    else:
        print(f"Erro ao buscar payment-slip-notification do billReceivableId {bill_receivable_id} e installmentId {installment_id}: {response.status_code}")
        print(response.text) 