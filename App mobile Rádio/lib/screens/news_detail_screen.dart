import 'package:flutter/material.dart';
import 'package:webview_flutter/webview_flutter.dart';
import 'package:provider/provider.dart';
import 'package:url_launcher/url_launcher.dart';
import '../services/theme_service.dart';
import '../utils/theme.dart';

class NewsDetailScreen extends StatefulWidget {
  final String newsUrl;
  final String newsTitle;
  
  const NewsDetailScreen({
    super.key,
    required this.newsUrl,
    required this.newsTitle,
  });

  @override
  State<NewsDetailScreen> createState() => _NewsDetailScreenState();
}

class _NewsDetailScreenState extends State<NewsDetailScreen> {
  late final WebViewController _controller;
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    
    // Configuração do WebView
    _controller = WebViewController()
      ..setJavaScriptMode(JavaScriptMode.unrestricted)
      ..setBackgroundColor(Colors.white)
      ..setNavigationDelegate(
        NavigationDelegate(
          onProgress: (int progress) {
            // Atualiza progresso de carregamento
          },
          onPageStarted: (String url) {
            setState(() {
              _isLoading = true;
            });
          },
          onPageFinished: (String url) {
            setState(() {
              _isLoading = false;
            });
          },
          onWebResourceError: (WebResourceError error) {
            debugPrint('Erro no WebView: ${error.description}');
          },
        ),
      )
      ..loadRequest(Uri.parse(widget.newsUrl));
  }

  @override
  Widget build(BuildContext context) {
    return Consumer<ThemeService>(
      builder: (context, themeService, child) {
        return Scaffold(
          backgroundColor: themeService.isDarkMode 
            ? AppTheme.darkBackground 
            : AppTheme.lightBackground,
          appBar: AppBar(
            backgroundColor: themeService.isDarkMode 
              ? AppTheme.darkSurface 
              : AppTheme.lightSurface,
            foregroundColor: themeService.isDarkMode 
              ? Colors.white 
              : AppTheme.lightTextPrimary,
            title: Text(
              widget.newsTitle.length > 30 
                ? '${widget.newsTitle.substring(0, 30)}...'
                : widget.newsTitle,
              style: TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.w600,
                color: themeService.isDarkMode 
                  ? Colors.white 
                  : AppTheme.lightTextPrimary,
              ),
            ),
            elevation: 0,
            actions: [
              // Botão para abrir no navegador
              IconButton(
                onPressed: () async {
                  final url = Uri.parse(widget.newsUrl);
                  if (await canLaunchUrl(url)) {
                    await launchUrl(
                      url,
                      mode: LaunchMode.externalApplication,
                    );
                  }
                },
                icon: const Icon(
                  Icons.open_in_browser,
                  color: AppTheme.primaryOrange,
                ),
                tooltip: 'Abrir no navegador',
              ),
            ],
          ),
          body: Stack(
            children: [
              // WebView
              WebViewWidget(controller: _controller),
              
              // Loading indicator
              if (_isLoading)
                Container(
                  color: themeService.isDarkMode 
                    ? AppTheme.darkBackground 
                    : AppTheme.lightBackground,
                  child: Center(
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        const CircularProgressIndicator(
                          valueColor: AlwaysStoppedAnimation<Color>(
                            AppTheme.primaryOrange,
                          ),
                        ),
                        const SizedBox(height: 16),
                        Text(
                          'Carregando notícia...',
                          style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                            color: themeService.isDarkMode 
                              ? AppTheme.darkTextSecondary 
                              : AppTheme.lightTextSecondary,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
            ],
          ),
        );
      },
    );
  }
}