//+------------------------------------------------------------------+
//|                                               SimpleDrawingEA.mq5 |
//|                        Copyright 2024, Modelo PPO Trader        |
//|                    Versão Profissional - Com Estimativas        |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, Modelo PPO Trader"
#property link      "https://www.mql5.com"
#property version   "8.00"
#property description "EA Profissional - 3 Linhas + Estimativas Automáticas"

// Parâmetros de entrada
input bool ShowDebugInfo = true;           // Mostrar informações de debug
input int RequestInterval = 3;             // Intervalo entre requests (segundos)
input bool UseEstimatedData = true;        // Usar dados estimados se servidor offline

// Variáveis globais para controle de estado (ANTI-PISCADA)
static string g_last_formations = "";
static string g_last_breakeven = "";
static string g_last_signal = "";
static string g_last_trend = "";
static datetime g_last_update = 0;
static bool g_server_online = false;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    Print("✅ [EA-PRO] SimpleDrawingEA Profissional com Estimativas iniciado");
    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    Print("🔄 [DEINIT] SimpleDrawingEA Profissional finalizado!");
    CleanupAllObjects();
}

//+------------------------------------------------------------------+
//| Função personalizada para substituir strings                     |
//+------------------------------------------------------------------+
string CustomStringReplace(string source, string search, string replace)
{
    string result = source;
    int pos = StringFind(result, search);
    while(pos >= 0)
    {
        result = StringSubstr(result, 0, pos) + replace + StringSubstr(result, pos + StringLen(search));
        pos = StringFind(result, search, pos + StringLen(replace));
    }
    return result;
}

//+------------------------------------------------------------------+
//| Função para limpar strings                                      |
//+------------------------------------------------------------------+
string CleanString(string str)
{
    string result = str;
    result = CustomStringReplace(result, "\"", "");
    result = CustomStringReplace(result, "'", "");
    result = CustomStringReplace(result, " ", "");
    result = CustomStringReplace(result, "\n", "");
    result = CustomStringReplace(result, "\r", "");
    return result;
}

//+------------------------------------------------------------------+
//| Extrair array do JSON                                           |
//+------------------------------------------------------------------+
string ExtractArrayFromJSON(string json, string key)
{
    string search1 = "\"" + key + "\":[";
    int start = StringFind(json, search1);
    if(start < 0) 
    {
        string search2 = "'" + key + "':[";
        start = StringFind(json, search2);
        if(start < 0) return "";
        start += StringLen(search2);
    }
    else
    {
        start += StringLen(search1);
    }
    
    int end = StringFind(json, "]", start);
    if(end < 0) return "";
    
    return StringSubstr(json, start, end - start);
}

//+------------------------------------------------------------------+
//| Extrair string do JSON                                          |
//+------------------------------------------------------------------+
string ExtractStringFromJSON(string json, string key)
{
    string search1 = "\"" + key + "\":\"";
    int start = StringFind(json, search1);
    if(start >= 0)
    {
        start += StringLen(search1);
        int end = StringFind(json, "\"", start);
        if(end >= 0) return StringSubstr(json, start, end - start);
    }
    
    string search2 = "'" + key + "':'";
    start = StringFind(json, search2);
    if(start >= 0)
    {
        start += StringLen(search2);
        int end = StringFind(json, "'", start);
        if(end >= 0) return StringSubstr(json, start, end - start);
    }
    
    string search3 = "\"" + key + "\":";
    start = StringFind(json, search3);
    if(start >= 0)
    {
        start += StringLen(search3);
        int end1 = StringFind(json, ",", start);
        int end2 = StringFind(json, "}", start);
        int end = (end1 > 0 && end1 < end2) ? end1 : end2;
        if(end >= 0)
        {
            string result = StringSubstr(json, start, end - start);
            return CleanString(result);
        }
    }
    
    return "";
}

//+------------------------------------------------------------------+
//| Função para verificar se deve atualizar (ANTI-PISCADA)          |
//+------------------------------------------------------------------+
bool ShouldUpdate(string current_data, string &last_data)
{
    if(current_data != last_data)
    {
        last_data = current_data;
        return true;
    }
    return false;
}

//+------------------------------------------------------------------+
//| Função para limpar objetos por prefixo                          |
//+------------------------------------------------------------------+
void ClearObjects(string prefix)
{
    int total = ObjectsTotal(0, -1, -1);
    for(int i = total - 1; i >= 0; i--)
    {
        string name = ObjectName(0, i, -1, -1);
        if(StringFind(name, prefix) == 0) 
        {
            ObjectDelete(0, name);
        }
    }
}

//+------------------------------------------------------------------+
//| ESTIMAR BREAKEVEN baseado na análise técnica                    |
//+------------------------------------------------------------------+
double EstimateBreakeven()
{
    double high_20 = iHigh(Symbol(), Period(), iHighest(Symbol(), Period(), MODE_HIGH, 20, 0));
    double low_20 = iLow(Symbol(), Period(), iLowest(Symbol(), Period(), MODE_LOW, 20, 0));
    double current_price = iClose(Symbol(), Period(), 0);
    
    // Breakeven = ponto médio entre máxima e mínima recente
    double breakeven = (high_20 + low_20) / 2;
    
    // Ajustar baseado na posição atual do preço
    if(current_price > breakeven)
    {
        breakeven = breakeven + (current_price - breakeven) * 0.3; // 30% mais próximo do preço atual
    }
    else
    {
        breakeven = breakeven - (breakeven - current_price) * 0.3;
    }
    
    return breakeven;
}

//+------------------------------------------------------------------+
//| ESTIMAR DECISÃO DO MODELO baseado na análise técnica            |
//+------------------------------------------------------------------+
string EstimateModelDecision()
{
    double current_price = iClose(Symbol(), Period(), 0);
    double prev_price = iClose(Symbol(), Period(), 1);
    double sma_5 = 0;
    
    // Calcular SMA 5
    for(int i = 0; i < 5; i++)
    {
        sma_5 += iClose(Symbol(), Period(), i);
    }
    sma_5 = sma_5 / 5;
    
    // Decisão baseada em momentum e posição relativa à média
    if(current_price > prev_price && current_price > sma_5)
    {
        return "BUY";
    }
    else if(current_price < prev_price && current_price < sma_5)
    {
        return "SELL";
    }
    else
    {
        return "HOLD";
    }
}

//+------------------------------------------------------------------+
//| 1. DESENHAR AS 3 LINHAS (TOPOS, FUNDOS, NECKLINE)              |
//+------------------------------------------------------------------+
void DrawThreeLines(string response)
{
    string formations_str = ExtractArrayFromJSON(response, "formations");
    
    // Se não tem dados do servidor, usar dados estimados
    if(StringLen(formations_str) == 0 || !g_server_online)
    {
        formations_str = "estimated_data";
    }
    
    // ANTI-PISCADA: Só atualizar se mudou
    if(!ShouldUpdate(formations_str, g_last_formations)) return;
    
    Print("✅ [3-LINES] Atualizando 3 linhas principais");
    
    // Limpar linhas anteriores
    ClearObjects("LINE_");
    
    // Encontrar pontos de máxima e mínima das últimas 20 barras
    double highest_1 = 0, highest_2 = 0;
    double lowest_1 = 999999, lowest_2 = 999999;
    datetime high_time_1 = 0, high_time_2 = 0;
    datetime low_time_1 = 0, low_time_2 = 0;
    
    for(int bar = 1; bar <= 20; bar++)
    {
        double high = iHigh(Symbol(), Period(), bar);
        double low = iLow(Symbol(), Period(), bar);
        datetime time = iTime(Symbol(), Period(), bar);
        
        if(high > highest_1)
        {
            highest_2 = highest_1;
            high_time_2 = high_time_1;
            highest_1 = high;
            high_time_1 = time;
        }
        else if(high > highest_2)
        {
            highest_2 = high;
            high_time_2 = time;
        }
        
        if(low < lowest_1)
        {
            lowest_2 = lowest_1;
            low_time_2 = low_time_1;
            lowest_1 = low;
            low_time_1 = time;
        }
        else if(low < lowest_2)
        {
            lowest_2 = low;
            low_time_2 = time;
        }
    }
    
    // LINHA 1: Resistência (conecta topos)
    if(ObjectFind(0, "LINE_Resistance") < 0)
    {
        ObjectCreate(0, "LINE_Resistance", OBJ_TREND, 0, high_time_1, highest_1, high_time_2, highest_2);
        ObjectSetInteger(0, "LINE_Resistance", OBJPROP_COLOR, clrRed);
        ObjectSetInteger(0, "LINE_Resistance", OBJPROP_WIDTH, 2);
        ObjectSetInteger(0, "LINE_Resistance", OBJPROP_STYLE, STYLE_SOLID);
        ObjectSetInteger(0, "LINE_Resistance", OBJPROP_RAY_RIGHT, true);
    }
    
    // LINHA 2: Suporte (conecta fundos)
    if(ObjectFind(0, "LINE_Support") < 0)
    {
        ObjectCreate(0, "LINE_Support", OBJ_TREND, 0, low_time_1, lowest_1, low_time_2, lowest_2);
        ObjectSetInteger(0, "LINE_Support", OBJPROP_COLOR, clrLimeGreen);
        ObjectSetInteger(0, "LINE_Support", OBJPROP_WIDTH, 2);
        ObjectSetInteger(0, "LINE_Support", OBJPROP_STYLE, STYLE_SOLID);
        ObjectSetInteger(0, "LINE_Support", OBJPROP_RAY_RIGHT, true);
    }
    
    // LINHA 3: Neckline TRACEJADA (linha do pescoço)
    double mid_price_1 = (highest_1 + lowest_1) / 2;
    double mid_price_2 = (highest_2 + lowest_2) / 2;
    datetime mid_time_1 = (high_time_1 + low_time_1) / 2;
    datetime mid_time_2 = (high_time_2 + low_time_2) / 2;
    
    if(ObjectFind(0, "LINE_Neckline") < 0)
    {
        ObjectCreate(0, "LINE_Neckline", OBJ_TREND, 0, mid_time_1, mid_price_1, mid_time_2, mid_price_2);
        ObjectSetInteger(0, "LINE_Neckline", OBJPROP_COLOR, clrYellow);
        ObjectSetInteger(0, "LINE_Neckline", OBJPROP_WIDTH, 2);
        ObjectSetInteger(0, "LINE_Neckline", OBJPROP_STYLE, STYLE_DASH); // TRACEJADA!
        ObjectSetInteger(0, "LINE_Neckline", OBJPROP_RAY_RIGHT, true);
    }
}

//+------------------------------------------------------------------+
//| 2. BREAKEVEN COM TEXTO (ESTIMADO OU REAL)                      |
//+------------------------------------------------------------------+
void DrawBreakeven(string response)
{
    string breakeven_str = ExtractStringFromJSON(response, "breakeven");
    double breakeven_price = 0;
    
    // Se não tem dados do servidor, estimar breakeven
    if(StringLen(breakeven_str) == 0 || !g_server_online)
    {
        breakeven_price = EstimateBreakeven();
        breakeven_str = DoubleToString(breakeven_price, 2);
        Print("📊 [BREAKEVEN] Estimado: ", breakeven_price);
    }
    else
    {
        breakeven_price = StringToDouble(breakeven_str);
    }
    
    if(breakeven_price <= 0) return;
    
    // ANTI-PISCADA
    if(!ShouldUpdate(breakeven_str, g_last_breakeven)) return;
    
    // Linha de breakeven
    if(ObjectFind(0, "BREAKEVEN_Line") >= 0) ObjectDelete(0, "BREAKEVEN_Line");
    ObjectCreate(0, "BREAKEVEN_Line", OBJ_HLINE, 0, 0, breakeven_price);
    ObjectSetInteger(0, "BREAKEVEN_Line", OBJPROP_COLOR, clrWhite);
    ObjectSetInteger(0, "BREAKEVEN_Line", OBJPROP_WIDTH, 2);
    ObjectSetInteger(0, "BREAKEVEN_Line", OBJPROP_STYLE, STYLE_DOT);
    
    // Texto do breakeven
    if(ObjectFind(0, "BREAKEVEN_Text") >= 0) ObjectDelete(0, "BREAKEVEN_Text");
    ObjectCreate(0, "BREAKEVEN_Text", OBJ_TEXT, 0, TimeCurrent(), breakeven_price + 5);
    string breakeven_text = "⚖️ BREAKEVEN @ " + DoubleToString(breakeven_price, 2);
    if(!g_server_online) breakeven_text += " (EST)";
    ObjectSetString(0, "BREAKEVEN_Text", OBJPROP_TEXT, breakeven_text);
    ObjectSetInteger(0, "BREAKEVEN_Text", OBJPROP_COLOR, clrWhite);
    ObjectSetInteger(0, "BREAKEVEN_Text", OBJPROP_FONTSIZE, 10);
    ObjectSetInteger(0, "BREAKEVEN_Text", OBJPROP_ANCHOR, ANCHOR_LEFT);
}

//+------------------------------------------------------------------+
//| 3. DECISÃO DO MODELO (ESTIMADA OU REAL)                        |
//+------------------------------------------------------------------+
void DrawModelDecision(string response)
{
    string signal = ExtractStringFromJSON(response, "signal");
    
    // Se não tem dados do servidor, estimar decisão
    if(StringLen(signal) == 0 || !g_server_online)
    {
        signal = EstimateModelDecision();
        Print("🤖 [DECISÃO] Estimada: ", signal);
    }
    
    // ANTI-PISCADA
    if(!ShouldUpdate(signal, g_last_signal)) return;
    
    double current_price = iClose(Symbol(), Period(), 0);
    string decision_text = "";
    color decision_color = clrGray;
    
    if(signal == "BUY" || signal == "COMPRA")
    {
        decision_text = "🟢 DECISÃO: COMPRA";
        decision_color = clrLimeGreen;
    }
    else if(signal == "SELL" || signal == "VENDA")
    {
        decision_text = "🔴 DECISÃO: VENDA";
        decision_color = clrRed;
    }
    else
    {
        decision_text = "⚪ DECISÃO: AGUARDAR";
        decision_color = clrYellow;
    }
    
    // Adicionar indicador de estimativa
    if(!g_server_online) decision_text += " (EST)";
    
    // Texto da decisão
    if(ObjectFind(0, "DECISION_Text") >= 0) ObjectDelete(0, "DECISION_Text");
    ObjectCreate(0, "DECISION_Text", OBJ_TEXT, 0, TimeCurrent(), current_price + 40);
    ObjectSetString(0, "DECISION_Text", OBJPROP_TEXT, decision_text);
    ObjectSetInteger(0, "DECISION_Text", OBJPROP_COLOR, decision_color);
    ObjectSetInteger(0, "DECISION_Text", OBJPROP_FONTSIZE, 12);
    ObjectSetInteger(0, "DECISION_Text", OBJPROP_ANCHOR, ANCHOR_LEFT);
}

//+------------------------------------------------------------------+
//| 4. FORMAÇÕES GRÁFICAS BASEADAS NO MODELO                       |
//+------------------------------------------------------------------+
void DrawGraphicalFormations(string response)
{
    string formations_str = ExtractArrayFromJSON(response, "formations");
    if(StringLen(formations_str) == 0) return;
    
    string formations[];
    int n_form = 0;
    
    // Parse das formações
    if(StringLen(formations_str) > 0)
    {
        CustomStringReplace(formations_str, " ", "");
        int start = 0;
        int pos = 0;
        
        while((pos = StringFind(formations_str, ",", start)) >= 0)
        {
            string value = StringSubstr(formations_str, start, pos - start);
            ArrayResize(formations, n_form + 1);
            formations[n_form] = value;
            n_form++;
            start = pos + 1;
        }
        
        if(start < StringLen(formations_str))
        {
            string value = StringSubstr(formations_str, start);
            ArrayResize(formations, n_form + 1);
            formations[n_form] = value;
            n_form++;
        }
    }
    
    if(n_form >= 7)
    {
        string formation_type = CleanString(formations[0]);
        
        datetime t1 = (datetime)StringToInteger(formations[1]);
        double p1 = StringToDouble(formations[2]);
        datetime t2 = (datetime)StringToInteger(formations[3]);
        double p2 = StringToDouble(formations[4]);
        datetime t3 = (datetime)StringToInteger(formations[5]);
        double p3 = StringToDouble(formations[6]);
        
        datetime current_time = TimeCurrent();
        
        // Limpar formações anteriores
        ClearObjects("FORMATION_");
        
        if(formation_type == "triangle")
        {
            // TRIÂNGULO
            ObjectCreate(0, "FORMATION_Triangle_1", OBJ_TREND, 0, t1, p1, t2, p2);
            ObjectSetInteger(0, "FORMATION_Triangle_1", OBJPROP_COLOR, clrGold);
            ObjectSetInteger(0, "FORMATION_Triangle_1", OBJPROP_WIDTH, 2);
            ObjectSetInteger(0, "FORMATION_Triangle_1", OBJPROP_STYLE, STYLE_SOLID);
            
            ObjectCreate(0, "FORMATION_Triangle_2", OBJ_TREND, 0, t2, p2, t3, p3);
            ObjectSetInteger(0, "FORMATION_Triangle_2", OBJPROP_COLOR, clrGold);
            ObjectSetInteger(0, "FORMATION_Triangle_2", OBJPROP_WIDTH, 2);
            ObjectSetInteger(0, "FORMATION_Triangle_2", OBJPROP_STYLE, STYLE_SOLID);
            
            // Texto da formação
            ObjectCreate(0, "FORMATION_Text", OBJ_TEXT, 0, current_time, p3 + 20);
            ObjectSetString(0, "FORMATION_Text", OBJPROP_TEXT, "📐 TRIÂNGULO");
            ObjectSetInteger(0, "FORMATION_Text", OBJPROP_COLOR, clrGold);
            ObjectSetInteger(0, "FORMATION_Text", OBJPROP_FONTSIZE, 10);
        }
        else if(formation_type == "double_bottom")
        {
            // FUNDO DUPLO
            ObjectCreate(0, "FORMATION_DB_1", OBJ_TREND, 0, t1, p1, t2, p2);
            ObjectSetInteger(0, "FORMATION_DB_1", OBJPROP_COLOR, clrLimeGreen);
            ObjectSetInteger(0, "FORMATION_DB_1", OBJPROP_WIDTH, 2);
            ObjectSetInteger(0, "FORMATION_DB_1", OBJPROP_STYLE, STYLE_DASHDOT);
            
            ObjectCreate(0, "FORMATION_DB_2", OBJ_TREND, 0, t2, p2, t3, p3);
            ObjectSetInteger(0, "FORMATION_DB_2", OBJPROP_COLOR, clrLimeGreen);
            ObjectSetInteger(0, "FORMATION_DB_2", OBJPROP_WIDTH, 2);
            ObjectSetInteger(0, "FORMATION_DB_2", OBJPROP_STYLE, STYLE_DASHDOT);
            
            ObjectCreate(0, "FORMATION_Text", OBJ_TEXT, 0, current_time, p3 + 20);
            ObjectSetString(0, "FORMATION_Text", OBJPROP_TEXT, "🔄 FUNDO DUPLO");
            ObjectSetInteger(0, "FORMATION_Text", OBJPROP_COLOR, clrLimeGreen);
            ObjectSetInteger(0, "FORMATION_Text", OBJPROP_FONTSIZE, 10);
        }
        else if(formation_type == "double_top")
        {
            // TOPO DUPLO
            ObjectCreate(0, "FORMATION_DT_1", OBJ_TREND, 0, t1, p1, t2, p2);
            ObjectSetInteger(0, "FORMATION_DT_1", OBJPROP_COLOR, clrRed);
            ObjectSetInteger(0, "FORMATION_DT_1", OBJPROP_WIDTH, 2);
            ObjectSetInteger(0, "FORMATION_DT_1", OBJPROP_STYLE, STYLE_DASHDOT);
            
            ObjectCreate(0, "FORMATION_DT_2", OBJ_TREND, 0, t2, p2, t3, p3);
            ObjectSetInteger(0, "FORMATION_DT_2", OBJPROP_COLOR, clrRed);
            ObjectSetInteger(0, "FORMATION_DT_2", OBJPROP_WIDTH, 2);
            ObjectSetInteger(0, "FORMATION_DT_2", OBJPROP_STYLE, STYLE_DASHDOT);
            
            ObjectCreate(0, "FORMATION_Text", OBJ_TEXT, 0, current_time, p3 + 20);
            ObjectSetString(0, "FORMATION_Text", OBJPROP_TEXT, "🔄 TOPO DUPLO");
            ObjectSetInteger(0, "FORMATION_Text", OBJPROP_COLOR, clrRed);
            ObjectSetInteger(0, "FORMATION_Text", OBJPROP_FONTSIZE, 10);
        }
        else if(formation_type == "flag")
        {
            // BANDEIRA
            ObjectCreate(0, "FORMATION_Flag_Pole", OBJ_TREND, 0, t1, p1, t2, p2);
            ObjectSetInteger(0, "FORMATION_Flag_Pole", OBJPROP_COLOR, clrBlue);
            ObjectSetInteger(0, "FORMATION_Flag_Pole", OBJPROP_WIDTH, 3);
            ObjectSetInteger(0, "FORMATION_Flag_Pole", OBJPROP_STYLE, STYLE_SOLID);
            
            ObjectCreate(0, "FORMATION_Flag_Body", OBJ_RECTANGLE, 0, t2, p2 - 10, t3, p3 + 10);
            ObjectSetInteger(0, "FORMATION_Flag_Body", OBJPROP_COLOR, clrYellow);
            ObjectSetInteger(0, "FORMATION_Flag_Body", OBJPROP_BACK, false);
            ObjectSetInteger(0, "FORMATION_Flag_Body", OBJPROP_FILL, true);
            ObjectSetInteger(0, "FORMATION_Flag_Body", OBJPROP_WIDTH, 1);
            
            ObjectCreate(0, "FORMATION_Text", OBJ_TEXT, 0, current_time, p3 + 20);
            ObjectSetString(0, "FORMATION_Text", OBJPROP_TEXT, "🏁 BANDEIRA");
            ObjectSetInteger(0, "FORMATION_Text", OBJPROP_COLOR, clrBlue);
            ObjectSetInteger(0, "FORMATION_Text", OBJPROP_FONTSIZE, 10);
        }
    }
}

//+------------------------------------------------------------------+
//| Função principal para desenhar análise profissional            |
//+------------------------------------------------------------------+
void DrawProfessionalAnalysis(char &result[])
{
    string response = CharArrayToString(result);
    
    if(StringLen(response) < 10) 
    {
        Print("❌ [EA-ERROR] Resposta muito curta: ", StringLen(response), " caracteres");
        g_server_online = false;
    }
    else
    {
        g_server_online = true;
    }
    
    Print("✅ [EA-PRO] Atualizando análise profissional... Server: ", g_server_online ? "ONLINE" : "OFFLINE");
    
    // Desenhar todos os elementos profissionais (com estimativas se necessário)
    DrawThreeLines(response);           // 1. As 3 linhas principais
    DrawBreakeven(response);            // 2. Breakeven com texto (estimado)
    DrawModelDecision(response);        // 3. Decisão do modelo (estimada)
    DrawGraphicalFormations(response);  // 4. Formações gráficas
    
    Print("✅ [EA-PRO] Análise profissional atualizada com sucesso");
}

//+------------------------------------------------------------------+
//| Função para obter análise técnica via Flask                     |
//+------------------------------------------------------------------+
void GetTechnicalAnalysis()
{
    string url = "http://127.0.0.1:5000/dados";
    
    char data[], result[];
    string headers = "Content-Type: application/json\r\n";
    
    int res = WebRequest("GET", url, headers, 5000, data, result, headers);
    
    if(res == -1)
    {
        Print("❌ [EA-ERROR] Servidor Flask offline - usando estimativas");
        g_server_online = false;
        // Desenhar com dados estimados
        char empty_result[1];
        DrawProfessionalAnalysis(empty_result);
        return;
    }
    
    if(res != 200)
    {
        Print("❌ [EA-ERROR] Erro HTTP: ", res, " - usando estimativas");
        g_server_online = false;
        char empty_result[1];
        DrawProfessionalAnalysis(empty_result);
        return;
    }
    
    g_server_online = true;
    DrawProfessionalAnalysis(result);
}

//+------------------------------------------------------------------+
//| Função para limpar todos os objetos                             |
//+------------------------------------------------------------------+
void CleanupAllObjects()
{
    string prefixes[] = {"LINE_", "BREAKEVEN_", "DECISION_", "FORMATION_"};
    
    for(int p = 0; p < ArraySize(prefixes); p++)
    {
        int total = ObjectsTotal(0, -1, -1);
        for(int i = total - 1; i >= 0; i--)
        {
            string name = ObjectName(0, i, -1, -1);
            if(StringFind(name, prefixes[p]) == 0) 
            {
                ObjectDelete(0, name);
            }
        }
    }
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    static datetime last_request = 0;
    
    if(TimeCurrent() - last_request < RequestInterval) return;
    last_request = TimeCurrent();
    
    GetTechnicalAnalysis();
} 

