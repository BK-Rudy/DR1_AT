def q3():
    def find_contact(list, name):

        for contact in list:
            if contact['name'] == name:
                return contact['phone']
        return None

    if __name__ == "__main__":
        list = [
            {'name': 'Bruno', 'phone': '99861-8305'},
            {'name': 'Raiane', 'phone': '99591-9447'},
            {'name': 'Cecília', 'phone': '99835-7745'},
            {'name': 'Fernando', 'phone': '99777-6666'},
            {'name': 'Jorge', 'phone': '99888-4444'},
        ]

        name = input("name of the contact: ")
        phone = find_contact(list, name)

        if phone:
            print(f"O contato do {name} é {phone}.")
        else:
            print(f"Contato {name} não encontrado.")

def q4():
    import random

    def binary_search(data, target_isbn):
        lower_index, upper_index = 0, len(data) - 1
        iterations = 0

        while lower_index <= upper_index:
            iterations += 1

            middle_index = (lower_index + upper_index) // 2

            if data[middle_index] == target_isbn:
                return middle_index, iterations
            elif data[middle_index] < target_isbn:
                lower_index = middle_index + 1
            else:
                upper_index = middle_index - 1
            
        return "ISBN não encotnrado.", iterations

    def linear_search(data, target_isbn):
        iterations = 0

        for i, value in enumerate(data):
            iterations += 1
            if value == target_isbn:
                return i, iterations
        
        return "ISBN não encontrado.", iterations

    isbn_list = sorted(random.randint(1000000000000, 9999999999999) for _ in range(100000))

    for i in range(1, 5):
        target_isbn = random.choice(isbn_list)

        binary_result, binary_iterations = binary_search(isbn_list, target_isbn)
        linear_result, linear_iterations = linear_search(isbn_list, target_isbn)

        print(f"O resultado da busca binária foi: {binary_result} em {binary_iterations} iterações.")
        print(f"O resultado da busca linear foi: {linear_result} em {linear_iterations} iterações.")

def q6():
    import logging
    import time
    import random

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')


    def bubble_sort(arr):
        n = len(arr)
        for i in range(n):
            for j in range(0, n - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
        return arr


    def measure_execution_time(data):
        start = time.time()
        result = bubble_sort(data)
        end = time.time()
        return result, end - start


    def main():
        array_1000 = [random.randint(0, 100000) for _ in range(1000)]
        _, time_1000 = measure_execution_time(array_1000[:])
        logging.info(f"Tempo para 1000 elementos: {time_1000:.4f}s")

        array_10000 = [random.randint(0, 100000) for _ in range(10000)]
        _, time_10000 = measure_execution_time(array_10000[:])
        logging.info(f"Tempo para 10000 elementos: {time_10000:.4f}s")

        print(f"Tempo para 1000 elementos: {time_1000:.4f}s")
        print(f"Tempo para 10000 elementos: {time_10000:.4f}s")

    if __name__ == '__main__':
        main()

def q7():
    list = [1, 2, 3, 4, 5, 6, 7, 8, 2]

    class HashTable:
        def __init__(self, size):
            self.size = size
            self.table = [[] for _ in range(size)]

        def _hash(self, key):
            return hash(key) % self.size
        
        def insert(self, value):
            index = self._hash(value)
            if value not in self.table[index]:
                self.table[index].append(value)

        def contains(self, value):
            index = self._hash(value)
            return value in self.table[index]
        
    def has_duplicates(lst):
        hashtable = HashTable(len(lst))
        for item in lst:
            if hashtable.contains(item):
                return True
            hashtable.insert(item)
        return False

    print("Há duplicatas?", has_duplicates(list))

def q8():
    def selection_sort(list):
        n = len(list)
        for i in range(n):
            highest_index = i
            for j in  range(i + 1, n):
                if list[j]["Pontos"] > list[highest_index]["Pontos"]:
                    highest_index = j

            list[i], list[highest_index] = list[highest_index], list[i]

    array = [
        {"Nome": "Bruno", "Pontos": 1200},
        {"Nome": "Raiane", "Pontos": 800},
        {"Nome": "Cecília", "Pontos": 1500},
        {"Nome": "Fernando", "Pontos": 950},
        {"Nome": "Jorge", "Pontos": 1350},
        {"Nome": "Roseli", "Pontos": 700},
        {"Nome": "Luana", "Pontos": 1450},
        {"Nome": "Erick", "Pontos": 1100},
        {"Nome": "Ricardo", "Pontos": 900},
    ]

    selection_sort(array)

    for player in array:
        print(f"{player['Nome']} - {player['Pontos']} pontos")
    
def q9():
    import time

    class HashTable:
        def __init__(self, size):
            self.size = size
            self.table = [[] for _ in range(size)]

        def _hash(self, key):
            return hash(key) % self.size

        def insert(self, key, value):
            index = self._hash(key)
            for item in self.table[index]:
                if item[0] == key:
                    item[1] = value
                    return
            self.table[index].append([key, value])

        def search(self, key):
            index = self._hash(key)
            for item in self.table[index]:
                if item[0] == key:
                    return item[1]
            return None

        def remove(self, key):
            index = self._hash(key)
            for item in self.table[index]:
                if item[0] == key:
                    self.table[index].remove(item)
                    return True
            return False


    class Profile:
        def __init__(self, username, details):
            self.username = username
            self.details = details


    class SequentialList:
        def __init__(self):
            self.list = []

        def insert(self, username, details):
            self.list.append(Profile(username, details))

        def retrieve(self, username):
            for profile in self.list:
                if profile.username == username:
                    return profile.details
            return None


    def main():
        data_size = 500_000
        user_data = [("user" + str(i), f"details{i}") for i in range(data_size)]
        target_user = "user999999"

        #HashTable
        hash_table = HashTable(size=10_000)

        start = time.time()
        for username, details in user_data:
            hash_table.insert(username, details)
        hash_insert_time = time.time() - start

        start = time.time()
        hash_table_result = hash_table.search(target_user)
        hash_search_time = time.time() - start

        print(f"Tempo de inserção na HashTable: {hash_insert_time:.6f}s")
        print(f"Tempo de busca na HashTable: {hash_search_time:.6f}s")
        print(f"Resultado da busca: {hash_table_result}\n")

        #Sequencial
        sequential_list = SequentialList()

        start = time.time()
        for username, details in user_data:
            sequential_list.insert(username, details)
        list_insert_time = time.time() - start

        start = time.time()
        list_result = sequential_list.retrieve(target_user)
        list_search_time = time.time() - start

        print(f"Tempo de inserção na Lista Sequencial: {list_insert_time:.6f}s")
        print(f"Tempo de busca na Lista Sequencial: {list_search_time:.6f}s")
        print(f"Resultado da busca: {list_result}\n")

        print(f"Tempo de inserção: HashTable é {'mais rápida' if hash_insert_time < list_insert_time else 'mais lenta'}")
        print(f"Tempo de busca: HashTable é {'mais rápida' if hash_search_time < list_search_time else 'mais lenta'}")

    if __name__ == "__main__":
        main()

def q10():
    from typing import List, Optional


    class Stack:
        def __init__(self):
            self.items: List[str] = []

        def is_empty(self) -> bool:
            return len(self.items) == 0

        def push(self, item: str) -> None:
            self.items.append(item)

        def pop(self) -> Optional[str]:
            if not self.is_empty():
                return self.items.pop()
            return None

        def peek(self) -> Optional[str]:
            if not self.is_empty():
                return self.items[-1]
            return None

        def size(self) -> int:
            return len(self.items)

        def get_items(self) -> List[str]:
            return self.items.copy()


    class BrowserNavigation:
        def __init__(self):
            self.back_stack = Stack()
            self.forward_stack = Stack()
            self.current_page: Optional[str] = None

        def visit_page(self, page: str) -> None:
            if self.current_page:
                self.back_stack.push(self.current_page)
            self.current_page = page
            self.forward_stack = Stack()
            print(f"Visitou: {self.current_page}")

        def go_back(self) -> Optional[str]:
            if self.back_stack.is_empty():
                print("Não há registros no histórico anterior.")
                return None
            self.forward_stack.push(self.current_page)
            self.current_page = self.back_stack.pop()
            print(f"De volta para: {self.current_page}")
            return self.current_page

        def go_forward(self) -> Optional[str]:
            if self.forward_stack.is_empty():
                print("Não há paginas para ir para.")
                return None
            self.back_stack.push(self.current_page)
            self.current_page = self.forward_stack.pop()
            print(f"Ir para: {self.current_page}")
            return self.current_page

        def current(self) -> Optional[str]:
            return self.current_page


    def main():
        browser = BrowserNavigation()

        browser.visit_page("https://www.google.com")
        browser.visit_page("https://www.chess.com/home")
        browser.visit_page("https://www.instagram.com")

        browser.go_back()
        browser.go_back()

        browser.go_forward()

        browser.visit_page("https://pt.wikipedia.org/wiki/Wikipédia:Página_principal")

        browser.go_back()
        browser.go_forward()

    if __name__ == "__main__":
        main()

def q11():
    class Queue:
        def __init__(self):
            self.queue = []

        def add_client(self, name):
            self.queue.append(name)
            print(f"Cliente {name} adicionado à fila.")

        def attend_client(self):
            if not self.queue:
                print("Não há clientes na fila.")
                return None
            client = self.queue.pop(0)
            print(f"Atendendo cliente: {client}")
            return client

        def queue_size(self):
            size = len(self.queue)
            print(f"Total de clientes aguardando atendimento: {size}")
            return size


    def simulate_client_service():
        queue = Queue()

        queue.add_client("Bruno")
        queue.add_client("Raiane")
        queue.add_client("Cecília")
        queue.attend_client()

        queue.queue_size()

        queue.add_client("Fernando")
        queue.attend_client()
        queue.attend_client()

        queue.queue_size()

        queue.attend_client()
        queue.attend_client()


    if __name__ == "__main__":
        simulate_client_service()

def q12():
    import os

    def show_files(dir):

        try:
            items = os.listdir(dir)
            for item in items:
                file_path = os.path.join(dir, item)

                if os.path.isdir(file_path):
                    # Continua a busca em subdiretórios
                    lista_arquivos(caminho_longo)
                else:
                    # Exibe apenas arquivos
                    print(f"[ARQUIVO] {caminho_longo}")

        except PermissionError:
            print(f"Permissão negada: {dir}")
        except FileNotFoundError:
            print(f"Diretório não encontrado: {dir}")
        except Exception as e:
            print(f"Ocorreu um erro ao acessar: {dir}")


    DIR = r"/home/bruno-rudy/Documentos/Bruno_Koiasqui_Rudy_DR2_AT"

    show_files(DIR)
    
def q13():
    def knapsack_solver(weights, values, capacity):

        n = len(weights)
        dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]

        for i in range(1, n + 1):
            for w in range(1, capacity + 1):
                if weights[i - 1] <= w:
                    dp[i][w] = max(values[i - 1] + dp[i - 1][w - weights[i - 1]], dp[i - 1][w])
                else:
                    dp[i][w] = dp[i - 1][w]
        
        max_value = dp[n][w]

        selected_itens = []
        w = capacity
        for i in range(n, 0, -1):
            if dp[i][w] != dp[i - 1][w]:
                selected_itens.append(i - 1) 
                w -= weights[i - 1]  

        selected_itens.reverse()

        return max_value, selected_itens

    weights = [1, 3, 4, 5]
    values = [10, 40, 50, 70]
    capacity = 8

    max_value, selected_itens = knapsack_solver(weights, values, capacity)
    print(f"Valor máximo que pode ser obtido: {max_value}")
    print(f"Itens selecionados (índices): {selected_itens}")
    print(f"Itens selecionados (pesos): {[weights[i] for i in selected_itens]}")
    print(f"Itens selecionados (valores): {[weights[i] for i in selected_itens]}")

def q14():
    class Node:
        def __init__(self, key):
            self.key = key
            self.left = None
            self.right = None

    class BST:
        def __init__(self):
            self.root = None

        def insert(self, root, key):
            if root is None:
                return Node(key)
            if key < root.key:
                root.left = self.insert(root.left, key)
            else:
                root.right = self.insert(root.right, key)
            return root

        def search(self, root, key):
            if root is None or root.key == key:
                return root
            if key < root.key:
                return self.search(root.left, key)
            return self.search(root.right, key)

        def inorder(self, root):
            if root:
                self.inorder(root.left)
                print(root.key, end=" ")
                self.inorder(root.right)

    list = [100,50,150,30,70,130,170]
    bst = BST()
    root = None
    for price in list:
        root = bst.insert(root, price)

    found_node = bst.search(root, 70)
    print("Preço de valor 70 encontrado" if found_node else "Preço de valor 70 não encontrado")

def q15():
    class Node:
        def __init__(self, key):
            self.key = key
            self.left = None
            self.right = None

    class BinarySearchTree:
        def __init__(self):
            self.root = None

        def insert(self, key):
            if self.root is None:
                self.root = Node(key)
            else:
                self._insert(self.root, key)

        def _insert(self, current_node, key):
            if key < current_node.key:
                if current_node.left is None:
                    current_node.left = Node(key)
                else:
                    self._insert(current_node.left, key)
            elif key > current_node.key:
                if current_node.right is None:
                    current_node.right = Node(key)
                else:
                    self._insert(current_node.right, key)

        def search(self, key):
            return self._search(self.root, key)
        
        def _search(self, current_node, key):
            if current_node is None:
                return False
            if current_node.key == key:
                return True
            elif key < current_node.key:
                return self._search(current_node.left, key)
            else:
                return self._search(current_node.right, key)
            
        def _navigate_right(self, current_node):
            if current_node.right is None:
                return current_node.key
            else:
                return self._navigate_right(current_node.right)
        
        def _navigate_left(self, current_node):
            if current_node.left is None:
                return current_node.key
            else:
                return self._navigate_left(current_node.left)
            
        def find_largest(self):
            return self._navigate_right(self.root)
            
        def find_smallest(self):
            return self._navigate_left(self.root)

    grades = [85, 70, 95, 60, 75, 90, 100]

    grade_tree = BinarySearchTree()

    for grade in grades:
        grade_tree.insert(grade)

    print(f"A maior nota na árvore é: {grade_tree.find_largest()}")
    print(f"A menor nota na árvore é: {grade_tree.find_smallest()}")

def q16():
    class Node:
        def __init__(self, key):
            self.key = key
            self.left = None
            self.right = None

    class BST:
        def __init__(self):
            self.root = None

        def insert(self, root, key):
            if root is None:
                return Node(key)
            if key < root.key:
                root.left = self.insert(root.left, key)
            else:
                root.right = self.insert(root.right, key)
            return root

        def search(self, root, key):
            if root is None or root.key == key:
                return root
            if key < root.key:
                return self.search(root.left, key)
            return self.search(root.right, key)

        def inorder(self, root):
            if root:
                self.inorder(root.left)
                print(root.key, end=" ")
                self.inorder(root.right)

    class BSTWithDelete(BST):
        def delete(self, root, key):
            if root is None:
                return root
            if key < root.key:
                root.left = self.delete(root.left, key)
            elif key > root.key:
                root.right = self.delete(root.right, key)
            else:
                if root.left is None:
                    return root.right
                elif root.right is None:
                    return root.left
                min_node = self.find_min_node(root.right)
                root.key = min_node.key
                root.right = self.delete(root.right, min_node.key)
            return root

        def find_min_node(self, root):
            current = root
            while current.left:
                current = current.left
            return current

    codes = [45, 25, 65, 20, 30, 60, 70]
    bst = BSTWithDelete()
    root = None
    for code in codes:
        root = bst.insert(root, code)

    for code in [20, 25, 45]:
        print(f"Removendo código: {code}")
        root = bst.delete(root, code)
        bst.inorder(root)
        print()

#Execução
if __name__ == "__main__":
    print("Escolha a questão para executar:")
    print("3 - Questão 3")
    print("4 - Questão 4")
    print("6 - Questão 6")
    print("7 - Questão 7")
    print("8 - Questão 8")
    print("9 - Questão 9")
    print("10 - Questão 10")
    print("11 - Questão 11")
    print("12 - Questão 12")
    print("13 - Questão 13")
    print("14 - Questão 14")
    print("15 - Questão 15")
    print("16 - Questão 16")

    selected = input("Digite o número da questão: ")

    functions = {
        "3": q3,
        "4": q4,
        "6": q6,
        "7": q7,
        "8": q8,
        "9": q9,
        "10": q10,
        "11": q11,
        "12": q12,
        "13": q13,
        "14": q14,
        "15": q15,
        "16": q16,
    }

    if selected in functions:
        functions[selected]()
    else:
        print("Opção inválida!")
