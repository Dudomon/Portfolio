import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:url_launcher/url_launcher.dart';
import '../widgets/sticky_player.dart';
import '../widgets/news_list.dart';
import '../widgets/podcast_section.dart';
import '../widgets/podcast_mini_player.dart';
import '../services/theme_service.dart';
import '../utils/theme.dart';
import 'contact_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  @override
  Widget build(BuildContext context) {
    return Consumer<ThemeService>(
      builder: (context, themeService, child) {
        return Scaffold(
          backgroundColor: themeService.isDarkMode 
            ? AppTheme.darkBackground 
            : AppTheme.lightBackground,
          body: Column(
            children: [
              // Conteúdo scrollável
              Expanded(
                child: CustomScrollView(
                  slivers: [
                    // Header fixo no topo
                    SliverToBoxAdapter(
                      child: SafeArea(
                        child: Container(
                          margin: const EdgeInsets.only(left: 8, right: 8, top: 16, bottom: 8),
                          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                          decoration: BoxDecoration(
                            gradient: LinearGradient(
                              colors: [
                                AppTheme.primaryOrange.withOpacity(0.1),
                                themeService.isDarkMode 
                                  ? AppTheme.darkSurface.withOpacity(0.8)
                                  : AppTheme.lightSurface.withOpacity(0.8),
                              ],
                              begin: Alignment.topLeft,
                              end: Alignment.bottomRight,
                            ),
                            borderRadius: BorderRadius.circular(20),
                            border: Border.all(
                              color: AppTheme.primaryOrange.withOpacity(0.3),
                              width: 1,
                            ),
                            boxShadow: [
                              BoxShadow(
                                color: AppTheme.primaryOrange.withOpacity(0.2),
                                blurRadius: 12,
                                offset: const Offset(0, 6),
                              ),
                            ],
                          ),
                          child: Row(
                            children: [
                              // Ícone de notícias
                              Container(
                                width: 40,
                                height: 40,
                                decoration: BoxDecoration(
                                  gradient: LinearGradient(
                                    colors: [
                                      AppTheme.primaryOrange,
                                      AppTheme.primaryOrange.withOpacity(0.7),
                                    ],
                                    begin: Alignment.topLeft,
                                    end: Alignment.bottomRight,
                                  ),
                                  borderRadius: BorderRadius.circular(12),
                                  boxShadow: [
                                    BoxShadow(
                                      color: AppTheme.primaryOrange.withOpacity(0.4),
                                      blurRadius: 6,
                                      offset: const Offset(0, 3),
                                    ),
                                  ],
                                ),
                                child: const Icon(
                                  Icons.newspaper_rounded,
                                  color: Colors.white,
                                  size: 22,
                                ),
                              ),
                              const SizedBox(width: 16),
                              // Título com rádio e central
                              Expanded(
                                child: Column(
                                  mainAxisAlignment: MainAxisAlignment.center,
                                  crossAxisAlignment: CrossAxisAlignment.center,
                                  children: [
                                    Text(
                                      'Entre Rios 105.5FM',
                                      style: Theme.of(context).textTheme.titleLarge?.copyWith(
                                        fontWeight: FontWeight.w500,
                                        letterSpacing: 0.3,
                                        color: AppTheme.primaryOrange.withOpacity(0.9),
                                      ),
                                    ),
                                    const SizedBox(height: 2),
                                    Text(
                                      'Central de Notícias',
                                      style: Theme.of(context).textTheme.titleLarge?.copyWith(
                                        fontWeight: FontWeight.bold,
                                        letterSpacing: 0.5,
                                        color: themeService.isDarkMode 
                                          ? Colors.white 
                                          : AppTheme.lightTextPrimary,
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                              // Toggle de tema
                              Container(
                                width: 40,
                                height: 40,
                                decoration: BoxDecoration(
                                  color: themeService.isDarkMode 
                                    ? AppTheme.darkCard.withOpacity(0.7)
                                    : AppTheme.lightCard.withOpacity(0.9),
                                  borderRadius: BorderRadius.circular(12),
                                  border: Border.all(
                                    color: AppTheme.primaryOrange.withOpacity(0.3),
                                    width: 1,
                                  ),
                                ),
                                child: PopupMenuButton<String>(
                                  onSelected: (value) {
                                    switch (value) {
                                      case 'theme':
                                        themeService.toggleTheme();
                                        break;
                                      case 'about':
                                        Navigator.push(
                                          context,
                                          MaterialPageRoute(
                                            builder: (context) => const ContactScreen(),
                                          ),
                                        );
                                        break;
                                      case 'contato':
                                        _openUrl('https://wa.me/4949991686860');
                                        break;
                                      case 'locutor':
                                        _openUrl('https://api.whatsapp.com/send?phone=554936470292');
                                        break;
                                      case 'instagram':
                                        _openUrl('https://www.instagram.com/entreriosfm105.5/');
                                        break;
                                    }
                                  },
                                  icon: const Icon(
                                    Icons.menu,
                                    size: 20,
                                    color: AppTheme.primaryOrange,
                                  ),
                                  padding: EdgeInsets.zero,
                                  itemBuilder: (BuildContext context) => [
                                    const PopupMenuItem<String>(
                                      value: 'about',
                                      child: Row(
                                        children: [
                                          Icon(
                                            Icons.info_outline,
                                            size: 18,
                                            color: AppTheme.primaryOrange,
                                          ),
                                          SizedBox(width: 8),
                                          Text(
                                            'Sobre & Contato',
                                            style: TextStyle(fontSize: 14),
                                          ),
                                        ],
                                      ),
                                    ),
                                    PopupMenuItem<String>(
                                      value: 'theme',
                                      child: Row(
                                        children: [
                                          Icon(
                                            themeService.isDarkMode
                                              ? Icons.light_mode_outlined
                                              : Icons.dark_mode_outlined,
                                            size: 18,
                                            color: AppTheme.primaryOrange,
                                          ),
                                          const SizedBox(width: 8),
                                          Text(
                                            'Tema: ${themeService.isDarkMode ? 'Claro' : 'Escuro'}',
                                            style: const TextStyle(fontSize: 14),
                                          ),
                                        ],
                                      ),
                                    ),
                                    const PopupMenuItem<String>(
                                      value: 'contato',
                                      child: Row(
                                        children: [
                                          Icon(
                                            Icons.business,
                                            size: 18,
                                            color: AppTheme.primaryOrange,
                                          ),
                                          SizedBox(width: 8),
                                          Text(
                                            'Contato Comercial',
                                            style: TextStyle(fontSize: 14),
                                          ),
                                        ],
                                      ),
                                    ),
                                    const PopupMenuItem<String>(
                                      value: 'locutor',
                                      child: Row(
                                        children: [
                                          Icon(
                                            Icons.mic,
                                            size: 18,
                                            color: AppTheme.primaryOrange,
                                          ),
                                          SizedBox(width: 8),
                                          Text(
                                            'Fale com o locutor!',
                                            style: TextStyle(fontSize: 14),
                                          ),
                                        ],
                                      ),
                                    ),
                                    const PopupMenuItem<String>(
                                      value: 'instagram',
                                      child: Row(
                                        children: [
                                          Icon(
                                            Icons.camera_alt,
                                            size: 18,
                                            color: AppTheme.primaryOrange,
                                          ),
                                          SizedBox(width: 8),
                                          Text(
                                            'Instagram da 105.5',
                                            style: TextStyle(fontSize: 14),
                                          ),
                                        ],
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                            ],
                          ),
                        ),
                      ),
                    ),
                    
                    // Carrossel de notícias
                    SliverToBoxAdapter(
                      child: SizedBox(
                        height: MediaQuery.of(context).size.height * 0.5,
                        child: const NewsList(),
                      ),
                    ),
                    
                    // Divider
                    SliverToBoxAdapter(
                      child: Column(
                        children: [
                          const SizedBox(height: 8),
                          Divider(
                            color: themeService.isDarkMode 
                              ? AppTheme.darkDivider 
                              : AppTheme.lightDivider,
                            thickness: 1,
                          ),
                          const SizedBox(height: 8),
                        ],
                      ),
                    ),
                    
                    // Seção de Podcasts (agora como lista scrollável)
                    const SliverFillRemaining(
                      hasScrollBody: false,
                      child: PodcastSection(),
                    ),
                  ],
                ),
              ),
              
              // Mini-player de podcast (quando ativo)
              const PodcastMiniPlayer(),
              
              // Player fixo no bottom
              const StickyPlayer(),
            ],
          ),
        );
      },
    );
  }

  // Função para abrir URLs externas
  Future<void> _openUrl(String url) async {
    try {
      await launchUrl(
        Uri.parse(url),
        mode: LaunchMode.externalApplication,
      );
    } catch (e) {
      debugPrint('Erro ao abrir URL: $e');
    }
  }

  // Função para abrir WhatsApp
  Future<void> _openWhatsApp(String phoneNumber, String message) async {
    final String encodedMessage = Uri.encodeComponent(message);
    final String whatsappUrl = 'https://wa.me/$phoneNumber?text=$encodedMessage';
    
    try {
      if (await canLaunchUrl(Uri.parse(whatsappUrl))) {
        await launchUrl(
          Uri.parse(whatsappUrl),
          mode: LaunchMode.externalApplication,
        );
      } else {
        debugPrint('Não foi possível abrir o WhatsApp');
      }
    } catch (e) {
      debugPrint('Erro ao abrir WhatsApp: $e');
    }
  }
}