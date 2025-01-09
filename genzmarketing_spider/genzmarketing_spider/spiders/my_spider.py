import scrapy


class MySpiderSpider(scrapy.Spider):
    name = "my_spider"
    allowed_domains = ["genzmarketing.xyz"]
    start_urls = ["https://genzmarketing.xyz"]

    def parse(self, response):
        pass
