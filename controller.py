from demo_ai_class.main import Jondomo
from demo_ai_class.data.source_ilik import (ilik_texts,ilik_labels,ilik_label_mapping)
from demo_ai_class.data.source_barysh import (barysh_texts, barysh_labels, barysh_label_mapping)
from demo_ai_class.data.source_tabysh import (tabysh_texts, tabysh_labels, tabysh_label_mapping)
from demo_ai_class.data.source_jatysh import (jatysh_texts, jatysh_labels, jatysh_label_mapping)
from demo_ai_class.data.source_chygysh import (chgysh_texts, chygysh_labels, chygysh_label_mapping)
from demo_ai_class.data.source_plural import (plural_texts, plural_labels, plural_label_mapping)

epoch = 20
ilik = Jondomo("Илик",ilik_texts, ilik_labels,ilik_label_mapping, epoch)
barysh = Jondomo("Барыш",barysh_texts, barysh_labels, barysh_label_mapping, epoch)
tabysh = Jondomo("Табыш",tabysh_texts, tabysh_labels, tabysh_label_mapping, epoch)
jatysh = Jondomo("Жатыш",jatysh_texts, jatysh_labels, jatysh_label_mapping, epoch)
chygysh = Jondomo("Чыгыш",chgysh_texts, chygysh_labels, chygysh_label_mapping, epoch)
plural = Jondomo("Көптүк ",plural_texts, plural_labels, plural_label_mapping, epoch)

while True:
    input_text = list(map(str, input(" ENTER : ").split()))
    ilik.get_predict(input_text)
    barysh.get_predict(input_text)
    tabysh.get_predict(input_text)
    jatysh.get_predict(input_text)
    chygysh.get_predict(input_text)
    plural.get_predict(input_text)
