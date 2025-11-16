class Podcast {
  final String id;
  final String title;
  final String description;
  final String audioUrl;
  final String imageUrl;
  final String duration;
  final DateTime publishedDate;
  final String category;

  Podcast({
    required this.id,
    required this.title,
    required this.description,
    required this.audioUrl,
    required this.imageUrl,
    required this.duration,
    required this.publishedDate,
    required this.category,
  });

  factory Podcast.fromJson(Map<String, dynamic> json) {
    return Podcast(
      id: json['id'] ?? '',
      title: json['title'] ?? '',
      description: json['description'] ?? '',
      audioUrl: json['audio_url'] ?? '',
      imageUrl: json['image_url'] ?? '',
      duration: json['duration'] ?? '',
      publishedDate: DateTime.parse(json['published_date'] ?? DateTime.now().toIso8601String()),
      category: json['category'] ?? 'Geral',
    );
  }

  String get formattedDate {
    final months = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 
                   'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'];
    return '${publishedDate.day} ${months[publishedDate.month - 1]} ${publishedDate.year}';
  }
}