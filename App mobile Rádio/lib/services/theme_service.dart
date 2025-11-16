import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

class ThemeService extends ChangeNotifier {
  static const String _themeKey = 'theme_mode';
  bool _isDarkMode = true; // Padrão escuro
  
  bool get isDarkMode => _isDarkMode;
  
  ThemeService() {
    _loadThemeFromPrefs();
  }
  
  /// Carrega a preferência de tema salva
  Future<void> _loadThemeFromPrefs() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      _isDarkMode = prefs.getBool(_themeKey) ?? true; // Padrão escuro
      notifyListeners();
    } catch (e) {
      // Falha ao carregar, mantém padrão escuro
      _isDarkMode = true;
    }
  }
  
  /// Alterna entre tema escuro e claro
  Future<void> toggleTheme() async {
    _isDarkMode = !_isDarkMode;
    notifyListeners();
    
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setBool(_themeKey, _isDarkMode);
    } catch (e) {
      // Falha ao salvar, mas mantém a mudança na memória
      debugPrint('Erro ao salvar preferência de tema: $e');
    }
  }
  
  /// Define o tema diretamente
  Future<void> setTheme(bool isDark) async {
    if (_isDarkMode == isDark) return;
    
    _isDarkMode = isDark;
    notifyListeners();
    
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setBool(_themeKey, _isDarkMode);
    } catch (e) {
      debugPrint('Erro ao salvar preferência de tema: $e');
    }
  }
}