class News {
  final String id;
  final String titulo;
  final String resumo;
  final String data;
  final String? imagem;
  final String fonte;
  final String creditoObrigatorio;
  final String? urlOriginal;

  News({
    required this.id,
    required this.titulo,
    required this.resumo,
    required this.data,
    this.imagem,
    required this.fonte,
    required this.creditoObrigatorio,
    this.urlOriginal,
  });

  factory News.fromJson(Map<String, dynamic> json) {
    return News(
      id: json['id'] ?? '',
      titulo: json['titulo'] ?? '',
      resumo: json['resumo'] ?? '',
      data: json['data'] ?? '',
      imagem: json['imagem'] ?? _extractFirstImageFromMidias(json['midias']),
      fonte: json['fonte'] ?? 'Fonte Desconhecida',
      creditoObrigatorio: json['credito_obrigatorio'] ?? 
          'Fonte: ${json['fonte'] ?? 'Fonte Desconhecida'}',
      urlOriginal: json['url_original'],
    );
  }

  static String? _extractFirstImageFromMidias(dynamic midias) {
    if (midias == null || midias is! List || midias.isEmpty) return null;
    
    for (var item in midias) {
      if (item is String && item.isNotEmpty) {
        // Se é uma string, assume que é uma URL de imagem
        return item;
      } else if (item is Map && item['url'] is String) {
        // Se é um objeto com campo 'url'
        return item['url'];
      }
    }
    return null;
  }

  String get newsUrl => urlOriginal ?? 'https://radioentrerios.com.br/wp-content/noticias/index.php?id=$id';

  String get sourceCategory {
    final lowerSource = fonte.toLowerCase();
    
    // Fontes municipais
    if (lowerSource.contains('prefeitura') || 
        lowerSource.contains('câmara') || 
        lowerSource.contains('municipal') ||
        lowerSource.contains('município') ||
        lowerSource.contains('palmitos') ||
        lowerSource.contains('local')) {
      return 'MUNICIPAL';
    }
    
    // Fontes nacionais
    if (lowerSource.contains('g1') ||
        lowerSource.contains('globo') ||
        lowerSource.contains('uol') ||
        lowerSource.contains('folha') ||
        lowerSource.contains('estadão') ||
        lowerSource.contains('veja') ||
        lowerSource.contains('brasil') ||
        lowerSource.contains('nacional') ||
        lowerSource.contains('agência brasil') ||
        lowerSource.contains('cnn') ||
        lowerSource.contains('band') ||
        lowerSource.contains('record')) {
      return 'NACIONAL';
    }
    
    // Por padrão, considera regional
    return 'REGIONAL';
  }
}