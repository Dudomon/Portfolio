import 'dart:async';
import 'dart:convert';
import 'package:http/http.dart' as http;

class ApiService {
  static const String baseUrl = 'https://radioentrerios.com.br';
  
  // Headers padrão
  static Map<String, String> get headers => {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
  };

  // GET request genérico
  static Future<dynamic> get(String endpoint) async {
    try {
      final url = Uri.parse('$baseUrl$endpoint');
      print('🌐 Fazendo request: $url');
      
      final response = await http.get(url, headers: headers).timeout(
        const Duration(seconds: 10),
      );
      
      print('📊 Status: ${response.statusCode}');
      print('📄 Body length: ${response.body.length}');
      
      if (response.statusCode == 200) {
        if (response.body.isEmpty) {
          throw Exception('Resposta vazia do servidor');
        }
        
        try {
          return json.decode(response.body);
        } catch (e) {
          print('💥 JSON inválido: ${response.body.substring(0, 200)}...');
          throw Exception('JSON inválido: $e');
        }
      } else {
        throw Exception('Erro ${response.statusCode}: ${response.body}');
      }
    } on TimeoutException {
      throw Exception('Timeout na conexão');
    } catch (e) {
      throw Exception('Erro de conexão: $e');
    }
  }

  // POST request genérico
  static Future<dynamic> post(String endpoint, Map<String, dynamic> data) async {
    try {
      final url = Uri.parse('$baseUrl$endpoint');
      final response = await http.post(
        url,
        headers: headers,
        body: json.encode(data),
      );
      
      if (response.statusCode == 200 || response.statusCode == 201) {
        return json.decode(response.body);
      } else {
        throw Exception('Erro ao enviar dados: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Erro de conexão: $e');
    }
  }
}