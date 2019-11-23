{%- extends 'markdown.tpl' -%}{% block body %}---
{% if resources.title != "" and resources.title != nil %}title: {{resources.title}}{% endif %}
{% if resources.author != "" and resources.author != nil %}author: {{resources.author}}{% endif %}
{% if resources.date != "" and resources.date != nil %}date: {{resources.date}}{% endif %}
{% if resources.tags != "" and resources.tags != nil %}tags: {{resources.tags}}{% endif %}
{% if resources.summary != "" and resources.summary != nil %}summary: "{{resources.summary}}"{% endif %}
---

{{ super() }}
{%- endblock body %}

{% block codecell -%}
<div class="codecell" markdown="1">
{{ super() }}
</div>
{% endblock codecell %}

{% block input_group -%}
<div class="input_area" markdown="1">
{{ super() }}
</div>
{% endblock input_group %}

{% block output_group -%}
<div class="output_area" markdown="1">
{{ super() }}
</div>
{% endblock output_group %}

