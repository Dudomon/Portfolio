import 'package:flutter/material.dart';
import 'package:cached_network_image/cached_network_image.dart';
import 'package:url_launcher/url_launcher.dart';
import 'package:provider/provider.dart';
import 'package:share_plus/share_plus.dart';
import '../models/news.dart';
import '../services/theme_service.dart';
import '../utils/theme.dart';

class NewsCard extends StatelessWidget {
  final News news;
  
  const NewsCard({
    super.key,
    required this.news,
  });

  @override
  Widget build(BuildContext context) {
    return Consumer<ThemeService>(
      builder: (context, themeService, child) {
        final cardColor = themeService.isDarkMode 
          ? AppTheme.darkSurface 
          : AppTheme.lightCard;
        
        return Container(
          margin: EdgeInsets.symmetric(
            horizontal: MediaQuery.of(context).size.width * 0.04,
            vertical: 8
          ),
          decoration: BoxDecoration(
            color: cardColor,
            borderRadius: BorderRadius.circular(16),
            border: Border.all(
              color: AppTheme.primaryOrange.withOpacity(0.3),
              width: 1.5,
            ),
            boxShadow: [
              BoxShadow(
                color: themeService.isDarkMode 
                  ? AppTheme.primaryOrange.withOpacity(0.2)
                  : Colors.black.withOpacity(0.15),
                blurRadius: 12,
                offset: const Offset(0, 6),
              ),
            ],
          ),
          child: Material(
            color: Colors.transparent,
            child: InkWell(
              onTap: () => _openNews(context),
              borderRadius: BorderRadius.circular(16),
              splashColor: AppTheme.primaryOrange.withOpacity(0.1),
              highlightColor: AppTheme.primaryOrange.withOpacity(0.05),
              child: Stack(
                children: [
                  Padding(
                    padding: const EdgeInsets.all(12),
                    child: SingleChildScrollView(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        mainAxisSize: MainAxisSize.min,
                        children: [
                    // Imagem da notícia (topo)
                    if (news.imagem != null) ...[
                      Container(
                        width: double.infinity,
                        height: MediaQuery.of(context).size.width < 600 
                          ? MediaQuery.of(context).size.height * 0.15 
                          : MediaQuery.of(context).size.height * 0.18,
                        decoration: BoxDecoration(
                          borderRadius: BorderRadius.circular(12),
                          border: Border.all(
                            color: AppTheme.primaryOrange.withOpacity(0.3),
                            width: 1,
                          ),
                        ),
                        child: ClipRRect(
                          borderRadius: BorderRadius.circular(11),
                          child: CachedNetworkImage(
                            imageUrl: 'https://radioentrerios.com.br/wp-content/noticias/image_proxy.php?url=${Uri.encodeComponent(news.imagem!)}',
                            width: double.infinity,
                            height: MediaQuery.of(context).size.height * 0.18,
                            fit: BoxFit.cover,
                            placeholder: (context, url) => Container(
                              decoration: BoxDecoration(
                                gradient: LinearGradient(
                                  colors: [
                                    AppTheme.primaryOrange.withOpacity(0.1),
                                    cardColor,
                                  ],
                                  begin: Alignment.topLeft,
                                  end: Alignment.bottomRight,
                                ),
                              ),
                              child: Center(
                                child: Column(
                                  mainAxisAlignment: MainAxisAlignment.center,
                                  children: [
                                    const CircularProgressIndicator(
                                      strokeWidth: 3,
                                      valueColor: AlwaysStoppedAnimation<Color>(
                                        AppTheme.primaryOrange,
                                      ),
                                    ),
                                    const SizedBox(height: 8),
                                    Text(
                                      'Carregando imagem...',
                                      style: Theme.of(context).textTheme.bodySmall?.copyWith(
                                        color: AppTheme.primaryOrange,
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                            ),
                            errorWidget: (context, url, error) => Container(
                              decoration: BoxDecoration(
                                gradient: LinearGradient(
                                  colors: [
                                    Colors.red.withOpacity(0.1),
                                    cardColor,
                                  ],
                                  begin: Alignment.topLeft,
                                  end: Alignment.bottomRight,
                                ),
                              ),
                              child: Center(
                                child: Column(
                                  mainAxisAlignment: MainAxisAlignment.center,
                                  children: [
                                    Icon(
                                      Icons.image_not_supported_outlined,
                                      color: Colors.red.withOpacity(0.7),
                                      size: 48,
                                    ),
                                    const SizedBox(height: 8),
                                    Text(
                                      'Imagem não disponível',
                                      style: Theme.of(context).textTheme.bodySmall?.copyWith(
                                        color: Colors.red.withOpacity(0.7),
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                            ),
                          ),
                        ),
                      ),
                      const SizedBox(height: 10),
                    ],
                    
                    // Conteúdo da notícia
                    Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        // Título
                        Text(
                          news.titulo,
                          style: Theme.of(context).textTheme.titleMedium?.copyWith(
                            fontWeight: FontWeight.bold,
                            height: 1.2,
                            fontSize: MediaQuery.of(context).size.width < 600 ? 14 : 16,
                          ),
                          maxLines: MediaQuery.of(context).size.width < 600 ? 3 : 2,
                          overflow: TextOverflow.ellipsis,
                        ),
                        const SizedBox(height: 8),
                          
                      ],
                    ),
                        ],
                      ),
                    ),
                  ),
                  
                  // Label da categoria da fonte no canto inferior esquerdo
                  Positioned(
                    bottom: 8,
                    left: 8,
                    child: Container(
                      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                      decoration: BoxDecoration(
                        color: Colors.transparent,
                        borderRadius: BorderRadius.circular(12),
                        border: Border.all(
                          color: AppTheme.primaryOrange,
                          width: 1,
                        ),
                      ),
                      child: Text(
                        news.sourceCategory,
                        style: const TextStyle(
                          color: AppTheme.primaryOrange,
                          fontSize: 10,
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                    ),
                  ),
                  
                  // Botão de compartilhar no canto inferior direito
                  Positioned(
                    bottom: 8,
                    right: 8,
                    child: Container(
                      width: 24,
                      height: 24,
                      decoration: BoxDecoration(
                        color: AppTheme.primaryOrange,
                        borderRadius: BorderRadius.circular(12),
                        boxShadow: [
                          BoxShadow(
                            color: AppTheme.primaryOrange.withOpacity(0.4),
                            blurRadius: 4,
                            offset: const Offset(0, 1),
                          ),
                        ],
                      ),
                      child: Material(
                        color: Colors.transparent,
                        child: InkWell(
                          onTap: () => _shareNews(),
                          borderRadius: BorderRadius.circular(12),
                          child: const Icon(
                            Icons.share_rounded,
                            color: Colors.white,
                            size: 14,
                          ),
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        );
      },
    );
  }

  Future<void> _openNews(BuildContext context) async {
    try {
      final url = Uri.parse(news.newsUrl);
      await launchUrl(
        url,
        mode: LaunchMode.externalApplication,
      );
    } catch (e) {
      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Erro ao abrir notícia: $e'),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }

  Future<void> _shareNews() async {
    await Share.share(
      '📰 ${news.titulo}\n\n'
      'Leia mais: ${news.newsUrl}\n\n'
      '🎵 Rádio Entre Rios 105.5 FM',
      subject: news.titulo,
    );
  }
}